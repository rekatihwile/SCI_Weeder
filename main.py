"""
LaserWeeder — main entry point.

Run:
    ./run_with_eli_venv.sh main.py

Quick reference — most-changed knobs live in config/ sub-modules:
  config/runtime_flags.py    — HOMING, FIRE, MOCK_GANTRY, FULL_AUTO, DETECTOR_MODE
  config/vision.py           — AI_CONFIDENCE, DEFAULT_MODEL, YOLO_DEVICE
  config/survey_params.py    — SURVEY_BURST_COUNT, SURVEY_CONFIDENCE_OVERRIDE, SURVEY_CROP_MODE
  config/alignment_params.py — FINE_ALIGN_KP_X/Y, FINE_ALIGN_DEADZONE_PX, LASER_FIRE_POWER
  config/hardware.py         — FRAME_WIDTH, WORKSPACE_X/Y_MAX, LASER_OFFSET_X_MM

Detection colours and label styles → vision/visualization.py (KNOBS at top of file)
"""
from pipeline.runtime import run_runtime


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LaserWeeder runtime")
    parser.add_argument(
        "--dry-run-grid-filter",
        action="store_true",
        help="run survey/triangulation/grid filtering with a mock gantry and stop before target execution",
    )
    args = parser.parse_args()

    try:
        run_runtime(
            use_real_gantry=not args.dry_run_grid_filter,
            execute_targets=not args.dry_run_grid_filter,
            dry_run_grid_filter=args.dry_run_grid_filter,
        )
        print("\n[MAIN] Runtime finished.")
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    except Exception as exc:
        print(f"\n[MAIN] Runtime failed: {exc}")
        raise


if __name__ == "__main__":
    main()
