# LaserWeeder GrandMaster Workspace

Cleaned runtime workspace for the diode-laser weed targeting system.

## Runtime flow

`main.py` is the main state machine:

1. initialize hardware
2. optionally home the gantry
3. run stereo survey burst
4. detect weed stem/keypoint candidates
5. stereo-match left/right detections
6. triangulate workspace targets
7. plan target order
8. coarse move
9. fine-align with pixel PD control
10. fire laser strike
11. repeat for remaining targets
12. close cameras/gantry and save trial recording if enabled

## Clean folder layout

```text
LaserWeeder_GrandMaster/
├── main.py
├── config.py
├── hardware_setup.py
├── control/
│   ├── coarse_move.py
│   ├── fine_align.py
│   ├── pixel_error_model.py
│   ├── calibration_correction.py
│   └── strike.py
├── hardware/
│   ├── cameras.py
│   └── gantry.py
├── vision/
│   ├── matching.py
│   └── detectors/
│       ├── ai_detector.py
│       └── manual_detector_local.py
├── planning/
│   └── target_planner.py
├── ui/
│   ├── terminal.py
│   ├── triangulation_debug.py
│   └── workspace_plot.py
├── data_collection/
│   ├── grid_capture.py
│   └── photo_dashboard.py
├── dev_tools/
│   └── calibration/
└── params/
    ├── cv_weights/
    ├── calibration/
    └── hardware/
```

## Setup

Create a fresh virtual environment outside version control:

```bash
cd LaserWeeder_GrandMaster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Do **not** copy an old `.venv` folder between machines. Recreate it using `requirements.txt`.

## Main run

```bash
python main.py
```

The main behavior is controlled through `config.py`. Important toggles:

- `DETECTOR_MODE = "ai"` or `"manual"`
- `TRIANGULATION_ONLY_MODE`
- `SHOW_TRIANGULATION_PLOT`
- `FIRE`
- `RECORD_TRIAL`
- `USE_PIXEL_ERROR_CORRECTION`

## Hardware utilities

Check camera IDs/properties:

```bash
python -m hardware.cameras --probe
python -m hardware.cameras --open
python -m hardware.cameras --open --view
```

Manually control the CNC/GRBL gantry:

```bash
python -m hardware.gantry
```

Available commands inside the gantry prompt:

```text
move X Y [feed]
raw GCODE
pos
home
quit
```

## Photo collection

`data_collection/grid_capture.py` merges the old full-workspace and scaled-workspace capture scripts.

Example:

```bash
python -m data_collection.grid_capture --scale 0.75
```

`--scale 1.0` covers the full configured workspace. Smaller values shrink the capture region about the workspace center.

## Params organization

- `params/cv_weights/`: YOLO and QPoint model weights
- `params/calibration/`: stereo calibration, rectification maps, pixel error model
- `params/hardware/`: camera and GRBL port configuration

`config.py` has been patched to point to these organized locations.

## What was intentionally removed

Removed from the clean workspace:

- legacy detector/camera/fine-align files
- old figure-generation `Photo_Tests`
- old trial video recordings
- unused app UI file
- top-level affine fitting scratch script
- Python `__pycache__` folders

Calibration tooling was kept under `dev_tools/calibration/` because it is useful when the physical stereo rig changes, but it is not part of the normal runtime loop.
