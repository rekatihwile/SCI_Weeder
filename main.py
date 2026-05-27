"""
LaserWeeder — main entry point.

Run:
    ./run_with_eli_venv.sh main.py

Quick reference — most-changed knobs live in config/ sub-modules:
  config/runtime_flags.py    — HOMING, FIRE, MOCK_GANTRY, FULL_AUTO, DETECTOR_MODE
    config/vision.py           — AI_CONFIDENCE, AI_CLASS_CONFIDENCE, TARGET_CLASSES, AVOID_CLASSES, DEFAULT_MODEL
    config/survey_params.py    — SURVEY_BURST_COUNT, SURVEY_CROP_MODE
  config/alignment_params.py — FINE_ALIGN_KP_X/Y, FINE_ALIGN_DEADZONE_PX, LASER_FIRE_POWER
  config/hardware.py         — FRAME_WIDTH, WORKSPACE_X/Y_MAX, LASER_OFFSET_X_MM
  config/experiment.py       — NUM_TRIALS
  config/scout.py            — SCOUT_ENABLED, SCOUT_ADVANCE_DISTANCE_M, SCOUT_ADVANCE_SPEED_MPS

Detection colours and label styles → vision/visualization.py (KNOBS at top of file)
"""

from pipeline.runtime import run_runtime, close_runtime_session


def main():
    import argparse
    from config import (
        NUM_TRIALS,
        SCOUT_ENABLED,
        SCOUT_MOVE_AFTER_TRIAL,
        SCOUT_REQUIRED_FOR_AUTO,
        SCOUT_CAN_INTERFACE,
        SCOUT_DRY_RUN,
        SCOUT_ADVANCE_DISTANCE_M,
        SCOUT_ADVANCE_SPEED_MPS,
    )

    parser = argparse.ArgumentParser(description="LaserWeeder runtime")
    parser.add_argument(
        "--dry-run-grid-filter",
        action="store_true",
        help="run survey/triangulation/grid filtering with a mock gantry and stop before target execution",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Scout setup
    # -------------------------------------------------------------------------
    scout = None
    if SCOUT_ENABLED:
        from scout.scout_controller import ScoutController
        scout = ScoutController(interface=SCOUT_CAN_INTERFACE, dry_run=SCOUT_DRY_RUN)
        print(f"[SCOUT] ScoutController created (interface={SCOUT_CAN_INTERFACE}, dry_run={SCOUT_DRY_RUN})")

        if SCOUT_REQUIRED_FOR_AUTO and NUM_TRIALS > 1:
            print("[SCOUT] Preflight check (SCOUT_REQUIRED_FOR_AUTO=True)...")
            conn = scout.check_connection()
            if not conn["ok"]:
                print(f"[SCOUT] PREFLIGHT FAILED: {conn['message']}")
                print("[SCOUT] Aborting — set SCOUT_REQUIRED_FOR_AUTO=False to skip Scout on failure.")
                scout.close()
                return
            print(f"[SCOUT] Preflight OK: {conn['message']}")

    # -------------------------------------------------------------------------
    # Shared session — keeps cameras / gantry / detector alive across trials
    # so they are not torn down and re-opened between runs.
    # -------------------------------------------------------------------------
    session = {}

    try:
        for trial_idx in range(1, NUM_TRIALS + 1):
            is_last_trial = (trial_idx >= NUM_TRIALS)

            if NUM_TRIALS > 1:
                print(f"\n[MAIN] ========== Trial {trial_idx} / {NUM_TRIALS} ==========")

            # Keep hardware open until the final trial so we avoid re-opening
            # cameras, re-loading the YOLO model, and re-enabling stepper hold.
            keep_open = not is_last_trial

            # Only pass scout when inter-trial movement should happen.
            scout_this_trial = (
                scout
                if (SCOUT_ENABLED and SCOUT_MOVE_AFTER_TRIAL and not is_last_trial)
                else None
            )

            trial_result = {}
            run_runtime(
                use_real_gantry=not args.dry_run_grid_filter,
                execute_targets=not args.dry_run_grid_filter,
                dry_run_grid_filter=args.dry_run_grid_filter,
                session=session,
                keep_resources_open=keep_open,
                trial_index=trial_idx,
                num_trials=NUM_TRIALS,
                result=trial_result,
                scout=scout_this_trial,
                is_last_trial=is_last_trial,
            )

            trial_status = trial_result.get("final_status", "unknown")
            if NUM_TRIALS > 1:
                print(f"[MAIN] Trial {trial_idx} status: {trial_status}")

            # If the trial did not complete cleanly, stop the loop.
            if trial_status not in ("complete", "grid_filter_dry_run"):
                if SCOUT_REQUIRED_FOR_AUTO and trial_result.get("scout_move_ok") is False:
                    print("[SCOUT] SCOUT_REQUIRED_FOR_AUTO=True — aborting trials after scout failure.")
                break

        if NUM_TRIALS > 1:
            print("\n[MAIN] All trials complete.")

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    finally:
        close_runtime_session(session)
        if scout is not None:
            backup_distance_m = -SCOUT_ADVANCE_DISTANCE_M * (NUM_TRIALS * 0.5 + 1)
            print(f"[SCOUT] Backing up {abs(backup_distance_m):.3f} m to return near start...")
            scout.move_forward(backup_distance_m, SCOUT_ADVANCE_SPEED_MPS)
        scout.close()

if __name__ == "__main__":
    main()

