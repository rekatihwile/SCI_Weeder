# Pre-refactor snapshot

Generated during modular cleanup.

## Repository

- Workspace: `/home/eli/LaserWeeder_CleanRuntime`
- Branch: `clean-runtime-v2`
- Commit: `9f77bb49c4e4c6bd16cad6096be70d31df4d27bf`
- Short HEAD: `9f77bb4 fixed bringup steps`

## Git status

```text
## clean-runtime-v2...origin/clean-runtime-v2
```

## Recent commits

```text
9f77bb4 (HEAD -> clean-runtime-v2, origin/clean-runtime-v2) fixed bringup steps
2dbf162 finally FUCKING WORKING
607f991 Add safe mock gantry and runtime preflight
cdfa513 (origin/grandmaster-cleanup, valid-jetson-headless-controll, grandmaster-cleanup) post processing video with time stamps and status bar added
5bf3a0d post processing video with time stamps and status bar added
```

## Main size

```text
683 main.py
```

## Current bringup scripts

```text
bringup/00_env_check.py
bringup/01_camera_open.py
bringup/02_yolo_detection.py
bringup/03_gantry_status.py
bringup/04_gantry_home.py
bringup/05_gantry_move_survey.py
bringup/06_survey_detect_only.py
bringup/07_match_plan_only.py
bringup/08_runtime_step_machine.py
bringup/_nms_patch.py
```

## Baseline compile

Passed:

```bash
./run_with_eli_venv.sh -m py_compile main.py config.py hardware/cameras.py hardware/gantry.py vision/detectors/ai_detector.py control/coarse_move.py vision/matching.py planning/target_planner.py
```

## Lab note

Lab lights may be off during validation, so low detection counts are not automatically a failure. Crashes, camera failures, serial failures, or impossible all-zero results under known-good lighting are the important failure signals.
