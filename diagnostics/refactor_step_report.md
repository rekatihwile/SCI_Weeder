# Modular cleanup step report

Generated after staged validation.

## Summary

The bringup ladder now shares reusable code from `pipeline/steps/`, and `main.py` is a small orchestrator that calls `pipeline.steps.runtime.run_runtime(use_real_gantry=True, execute_targets=True)`.

The refactor was validated after each meaningful extraction with compile checks, the relevant bringup script, and `main.py` where requested. No validation run ended in a Python crash.

## Files created

```text
diagnostics/main_before_small_orchestrator.py
diagnostics/pre_refactor_snapshot.md
diagnostics/refactor_step_report.md
pipeline/steps/__init__.py
pipeline/steps/context.py
pipeline/steps/detector_setup.py
pipeline/steps/camera_setup.py
pipeline/steps/gantry_setup.py
pipeline/steps/survey.py
pipeline/steps/match_plan.py
pipeline/steps/runtime.py
```

## Files modified

```text
bringup/01_camera_open.py
bringup/02_yolo_detection.py
bringup/04_gantry_home.py
bringup/05_gantry_move_survey.py
bringup/06_survey_detect_only.py
bringup/07_match_plan_only.py
bringup/08_runtime_step_machine.py
main.py
```

Validation also updated runtime outputs/log artifacts:

```text
bringup/logs/01_left.jpg
bringup/logs/01_right.jpg
bringup/logs/02_left.jpg
bringup/logs/02_right.jpg
bringup/logs/06_left.jpg
bringup/logs/06_right.jpg
experiments/metrics/run_summary.csv
experiments/metrics/target_summary.csv
planning/actual_pd_targets.json
planning/predicted_workspace_targets.json
trial_recordings/
experiments/metrics/json/
```

## Main.py size

- Pre-refactor snapshot: `683 main.py`
- Immediately before small orchestrator backup: `669 diagnostics/main_before_small_orchestrator.py`
- New orchestrator: `16 main.py`

## Step validations

### Step 0 snapshot

Passed:

```bash
./run_with_eli_venv.sh -m py_compile main.py config.py hardware/cameras.py hardware/gantry.py vision/detectors/ai_detector.py control/coarse_move.py vision/matching.py planning/target_planner.py
```

Snapshot saved to:

```text
diagnostics/pre_refactor_snapshot.md
```

### Step 2 detector setup

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/detector_setup.py bringup/02_yolo_detection.py main.py
./run_with_eli_venv.sh bringup/02_yolo_detection.py | tee bringup/logs/02_after_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_detector_step_extract.log
```

Result highlights:

```text
02_after_step_extract.log: RESULT: PASS (left=2 right=2 detections)
main_after_detector_step_extract.log: Status complete
```

### Step 3 camera setup

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/camera_setup.py bringup/01_camera_open.py bringup/02_yolo_detection.py main.py
./run_with_eli_venv.sh bringup/01_camera_open.py | tee bringup/logs/01_after_camera_step_extract.log
./run_with_eli_venv.sh bringup/02_yolo_detection.py | tee bringup/logs/02_after_camera_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_camera_step_extract.log
```

Result highlights:

```text
01_after_camera_step_extract.log: RESULT: PASS (30/30 pairs)
02_after_camera_step_extract.log: RESULT: PASS (left=0 right=1 detections)
main_after_camera_step_extract.log: Status complete
```

`01` needed camera attempt 2 once. That matches the known bounded camera-open recovery path.

### Step 4 gantry setup

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/gantry_setup.py bringup/03_gantry_status.py bringup/04_gantry_home.py bringup/05_gantry_move_survey.py main.py
./run_with_eli_venv.sh bringup/03_gantry_status.py | tee bringup/logs/03_after_gantry_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_gantry_step_extract.log
```

Result highlights:

```text
03_after_gantry_step_extract.log: RESULT: PASS (serial opened and status response received)
main_after_gantry_step_extract.log: Status complete
```

`04` and `05` were compiled but not automatically run, preserving their typed motion confirmations.

### Step 5 survey extraction

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/survey.py bringup/06_survey_detect_only.py main.py
./run_with_eli_venv.sh bringup/06_survey_detect_only.py | tee bringup/logs/06_after_survey_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_survey_step_extract.log
```

Result highlights:

```text
06_after_survey_step_extract.log: RESULT: PASS (left=4 right=2 detections)
main_after_survey_step_extract.log: Status complete
```

`06` needed camera attempt 2 once and recovered.

### Step 6 match/plan extraction

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/match_plan.py bringup/07_match_plan_only.py main.py
./run_with_eli_venv.sh bringup/07_match_plan_only.py | tee bringup/logs/07_after_match_plan_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_match_plan_step_extract.log
```

Result highlights:

```text
07_after_match_plan_step_extract.log: RESULT: PASS (matched=1 solved=1 planned=1)
main_after_match_plan_step_extract.log: Status complete
```

Note: `main.py` currently prints the match summary twice because the old accept/rescan state still performs matching before the shared helper performs match/triangulate/plan. This is cosmetic and did not break validation.

### Step 7 runtime wrapper for 08

Passed:

```bash
./run_with_eli_venv.sh -m py_compile pipeline/steps/runtime.py bringup/08_runtime_step_machine.py main.py
./run_with_eli_venv.sh bringup/08_runtime_step_machine.py | tee bringup/logs/08_after_runtime_step_extract.log
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_runtime_step_extract.log
```

Result highlights:

```text
08_after_runtime_step_extract.log: RESULT: PASS (pipeline reached PLAN step; planned=0)
main_after_runtime_step_extract.log: Status complete
```

### Step 8 small main.py

Passed:

```bash
./run_with_eli_venv.sh -m py_compile main.py pipeline/steps/*.py
./run_with_eli_venv.sh bringup/01_camera_open.py
./run_with_eli_venv.sh bringup/02_yolo_detection.py
./run_with_eli_venv.sh bringup/06_survey_detect_only.py
./run_with_eli_venv.sh bringup/07_match_plan_only.py
./run_with_eli_venv.sh bringup/08_runtime_step_machine.py
./run_with_eli_venv.sh main.py | tee diagnostics/main_after_small_orchestrator.log
```

Result highlights:

```text
01_camera_open.py: RESULT: PASS (30/30 pairs)
02_yolo_detection.py: RESULT: PASS (left=3 right=0 detections)
06_survey_detect_only.py: RESULT: PASS (left=1 right=0 detections)
07_match_plan_only.py: RESULT: PASS (matched=0 solved=0 planned=0)
08_runtime_step_machine.py: exited cleanly with MockGantry and no real motion
main_after_small_orchestrator.log: Status complete
main_after_small_orchestrator.log: [MAIN] Runtime finished.
```

`07` needed all three bounded camera-open attempts before succeeding. It recovered without manual intervention.

## Failures

No validation command failed with a nonzero exit or Python traceback after the relevant fixes.

Observed non-fatal issues:

- Low and asymmetric detections occurred repeatedly. This is probably due to lab lighting/scene conditions, not the refactor.
- Fine-align/re-ID often found no stable stereo candidates, so `main.py` completed with `0` targets fired. `FIRE=False`, and the runtime safely skipped strikes.
- `main.py` currently prints duplicated match summaries after Step 6 because matching is still used for survey acceptance and again inside the shared match/plan helper.
- The known camera-open issue still appears intermittently, but the bounded retry path recovered in these runs.

## Data collection readiness

The software plumbing is validated enough for cautious runtime testing: bringup scripts and the small `main.py` orchestrator all ran successfully.

For meaningful real data collection, restore/verify lighting and scene setup first. The latest runs completed safely, but low detections and failed local re-ID mean the current visual setup is not yet producing strong target-lock data.
