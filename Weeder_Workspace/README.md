# LASER WEEDER — DEBUG / OPERATION README

This README is for operating and debugging the modular workspace as it exists **after the config cleanup**.  
It is not meant to explain the full project from scratch. It is meant to answer:

- what `main.py` does
- where the important values live
- what to change when behavior is wrong
- how to diagnose the most common failures quickly

---
# In your terminal
cd ~/Documents/oxnard_test
tar cf Weeder_Workspace.tar Weeder_Workspace/


# 1. QUICK START

## Normal run

Run:

```bash
python main.py
```

Expected high-level sequence:

1. Open cameras
2. Home gantry
3. Move to survey position
4. Run global survey
5. Match left/right detections
6. Plan workspace targets
7. For each target:
   - coarse move
   - fine align
   - strike

If the system does not behave like that, something is wrong in either:

- startup / hardware init
- survey detection
- matching
- fine alignment
- strike execution

---

# 2. WHAT `main.py` ACTUALLY DOES

`main.py` is the state machine. It drives the whole run.

## State order

The main states are:

- `INIT`
- `HOME`
- `SURVEY`
- `SURVEY_CONFIRM`
- `DETECT`
- `MATCH`
- `PLAN`
- `EXECUTE`
- `DONE`

## What happens in each state

### `INIT`

Creates:

- `Gantry`
- `StereoCameras`
- detector (`AIDetector` or `ManualDetectorLocal`)
- `TriangulationCoarseMover`

Also clears the actual-target log for the run.

### `HOME`

- opens cameras
- homes gantry

### `SURVEY`

- moves to the survey position from `config.py`

### `SURVEY_CONFIRM`

Waits for user input before running the survey.

### `DETECT`

Calls the coarse mover’s survey detection pipeline:

- burst capture
- detection on each frame
- stability filtering

### `MATCH`

Matches stable left/right detections into stereo targets.

### `PLAN`

- triangulates targets into machine coordinates
- saves predicted targets
- optionally shows plots/debug images

### `EXECUTE`

For each target:

1. move to coarse XY
2. run fine alignment
3. if fine align succeeds, fire
4. if it fails, skip that target

---

# 3. THE MOST IMPORTANT CHANGE: KNOBS ARE NOW IN `config.py`

Before, important values were spread across multiple files.  
Now the main runtime tuning values are centralized in `config.py`.

That means your **first debugging stop should now be `config.py`**.

## Main sections in `config.py`

### Detector / AI

These control live AI detection behavior:

```python
AI_DISPLAY_SCALE
AI_BURST_SIZE
AI_MIN_STABLE_VIEWS
AI_CONFIDENCE
AI_IOM_THRESHOLD
```

### Global survey

These control burst survey behavior:

```python
SURVEY_BURST_COUNT
SURVEY_MIN_HITS
SURVEY_CLUSTER_RADIUS_PX
```

### Fine align

These control the PD stage:

```python
FINE_ALIGN_CROP_SCALE
FINE_ALIGN_LK_WIN_SIZE
FINE_ALIGN_LK_MAX_LEVEL

FINE_ALIGN_KP_X
FINE_ALIGN_KD_X
FINE_ALIGN_KP_Y
FINE_ALIGN_KD_Y

FINE_ALIGN_STEP_MM
FINE_ALIGN_DEADZONE_PX
FINE_ALIGN_MAX_JOG_MM
FINE_ALIGN_FEED

FINE_ALIGN_BURST_COUNT
FINE_ALIGN_MIN_HITS
FINE_ALIGN_CLUSTER_RADIUS_PX

FINE_ALIGN_MAX_TIME_SEC
FINE_ALIGN_SETTLE_FRAMES
```

### Strike / laser

These control the strike pulse:

```python
LASER_FIRE_POWER
LASER_FIRE_DURATION_SEC
LASER_ARM_DELAY_SEC
LASER_TRIGGER_FEED
```

### Machine / survey geometry

These control coarse positioning behavior:

```python
SURVEY_POS_X
SURVEY_POS_Y

LASER_OFFSET_X_MM
LASER_OFFSET_Y_MM

TRI_SIGN_X
TRI_SIGN_Y
TRI_X_GAIN
TRI_Y_GAIN
```

---

# 4. FILES YOU WILL ACTUALLY TOUCH

These are the files that matter most during operation.

## `main.py`

Use this to understand:

- state flow
- when survey runs
- when matching runs
- when fine align happens
- when strike happens

If behavior order seems wrong, look here first.

---

## `config.py`

This is now the main tuning file.

Change this first when you want to adjust:

- confidence
- survey aggressiveness
- PD gains
- crop size
- timeout
- settle frames
- strike power/duration

---

## `control/coarse_move.py`

This handles:

- survey burst processing
- triangulation
- conversion from pixel target to machine XY
- workspace target saving

Look here if:

- triangulated points are obviously wrong
- the machine moves to nonsense coordinates
- survey detections exist but planning looks wrong

---

## `control/fine_align.py`

This handles:

- selecting local fine-align target
- crop creation
- LK optical flow tracking
- PD correction
- timeout / settle logic

Look here if:

- coarse move is okay, but final centering is bad
- the point drifts
- it times out
- it says “fine align failed”

---

## `vision/detectors/ai_detector.py`

This handles:

- YOLO detection
- q-point refinement
- burst-stable point generation for AI mode

Look here if:

- detections seem too sparse
- model works in one place but not another
- confidence / overlap behavior seems wrong

---

## `hardware/gantry.py`

This handles:

- serial communication
- homing
- absolute moves
- jog moves
- stepper hold release on shutdown
- laser command sending

Look here if:

- the gantry does not move right
- the motors stay locked after exit
- laser commands do not behave correctly

---

## `hardware/cameras.py`

This handles:

- opening both cameras
- applying camera settings
- flipping frames
- left/right startup check

Look here if:

- the left/right order is wrong
- the preview is wrong
- one camera does not open
- startup says the order is inconclusive

---

# 5. HOW TO TUNE THE SYSTEM NOW

Because the main values are centralized, tuning should be done in a specific order.

## If survey finds nothing

Start here in `config.py`:

```python
AI_CONFIDENCE
SURVEY_BURST_COUNT
SURVEY_MIN_HITS
SURVEY_CLUSTER_RADIUS_PX
```

Recommended direction:

- lower `AI_CONFIDENCE`
- lower `SURVEY_MIN_HITS`
- increase `SURVEY_CLUSTER_RADIUS_PX`

Typical interpretation:

- detections exist, but stable points = 0  
  → survey filter too strict
- stable points exist, but matched targets = 0  
  → matching/geometry issue

---

## If fine align keeps failing

Start here:

```python
FINE_ALIGN_MAX_TIME_SEC
FINE_ALIGN_SETTLE_FRAMES
FINE_ALIGN_DEADZONE_PX
FINE_ALIGN_KP_X
FINE_ALIGN_KP_Y
FINE_ALIGN_KD_X
FINE_ALIGN_KD_Y
FINE_ALIGN_CROP_SCALE
```

Recommended direction:

- increase `FINE_ALIGN_MAX_TIME_SEC` if timing out
- reduce `FINE_ALIGN_SETTLE_FRAMES` if it almost locks but never commits
- increase `FINE_ALIGN_DEADZONE_PX` if it hunts around center
- reduce gains if motion is too aggressive
- increase crop scale if the target leaves the crop too easily

---

## If the gantry overreacts during PD

Start here:

```python
FINE_ALIGN_KP_X
FINE_ALIGN_KD_X
FINE_ALIGN_KP_Y
FINE_ALIGN_KD_Y
FINE_ALIGN_STEP_MM
FINE_ALIGN_MAX_JOG_MM
```

Symptoms:

- overshoot
- oscillation
- point moves past center repeatedly

Fix direction:

- lower KP
- lower step size
- lower max jog
- increase deadzone slightly if noise is the problem

---

## If the laser fires too weakly or too strongly

Start here:

```python
LASER_FIRE_POWER
LASER_FIRE_DURATION_SEC
LASER_ARM_DELAY_SEC
```

Important:

- test with low power first
- short duration is safer than long duration
- do not jump to large durations

---

# 6. COMMON FAILURE MODES AND WHAT THEY USUALLY MEAN

## Problem: model sees plants in tuner, but survey finds 0 stable points

Most likely:

- confidence is okay
- stability filtering is too strict

Check:

```python
SURVEY_MIN_HITS
SURVEY_CLUSTER_RADIUS_PX
AI_CONFIDENCE
```

This exact pattern showed up in your old logs: detections existed, but the stability stage reduced them to zero.

---

## Problem: survey works, but matched targets = 0

Most likely:

- left/right matching is failing
- disparity / geometry is inconsistent
- camera order or calibration may be wrong

Check:

- `vision/matching.py`
- `hardware/cameras.py`
- calibration files in config:
  - `CALIB_NPZ_PATH`
  - `RECT_NPZ_PATH`

---

## Problem: fine align moves, but then says failed

Most likely:

- no valid local target chosen
- LK lost the point
- point left the crop
- timeout

Check:

```python
FINE_ALIGN_MAX_TIME_SEC
FINE_ALIGN_CROP_SCALE
FINE_ALIGN_LK_WIN_SIZE
FINE_ALIGN_LK_MAX_LEVEL
FINE_ALIGN_DEADZONE_PX
```

Also watch the terminal output.  
`fine_align.py` now prints more specific failure reasons than before.

---

## Problem: strike step is reached, but nothing fires

Most likely causes:

- fire sequence did not execute correctly
- strike settings are too weak
- gantry/laser command path is broken

Check:

- `control/strike/strike_patterns.py`
- `hardware/gantry.py`
- `LASER_FIRE_POWER`
- `LASER_FIRE_DURATION_SEC`

A previous real failure from the logs was:

- strike crashed because `Gantry` had no `_send_and_wait` method, so the laser never actually fired. fileciteturn1file3turn1file0

---

## Problem: motors stay locked after program exits

Most likely:

- GRBL stepper hold setting changed back too late
- board needed more time before serial close
- release only became effective after reconnecting in LaserGRBL

Check:

- `hardware/gantry.py`
- shutdown path
- step hold / soft reset logic

This also showed up in the logs and is a known real issue in this workspace.

---

## Problem: calibration script runs, but stereo quality is poor

Not all successful detections are good calibration views.

Common causes:

- too many duplicate images
- too many extreme edge views
- blurry frames
- weak geometry coverage
- fisheye stereo solve becomes unstable

This showed up in the old calibration/debug logs too.

---

# 7. DEBUGGING ORDER THAT ACTUALLY MAKES SENSE

When something breaks, debug in this order:

## Step 1: confirm startup

Do you see:

- camera open messages
- gantry home
- no immediate serial/camera errors

If not, do not debug CV yet.

## Step 2: confirm survey detections

Ask:

- do detections exist?
- do stable points exist?

If detections exist but stable points do not:

- tune survey filter, not the model first

## Step 3: confirm matching

Ask:

- are there matched stereo targets?

If not:

- check camera ordering
- check calibration
- check matching constraints

## Step 4: confirm coarse move

Ask:

- does the gantry move near the right plant?

If not:

- triangulation / calibration / signs / offsets are wrong

## Step 5: confirm fine align

Ask:

- does the local target stay visible?
- does the point converge?
- does it timeout?
- does it oscillate?

Tune fine-align parameters only after coarse move is believable.

## Step 6: confirm strike

Ask:

- did the system reach the strike step?
- did the command execute?
- was the power high enough to matter?

---

# 8. EXPECTED CONSOLE BEHAVIOR

A healthy run should look roughly like this:

1. camera startup messages
2. homing message
3. survey prompt
4. burst survey output
5. match summary
6. workspace plan
7. for each target:
   - current target print
   - coarse move
   - fine align live output
   - target result
   - strike

If the ordering is weird, inspect `main.py`.

---

# 9. WHAT CHANGED IN THIS “IMPROVED” WORKSPACE

The main improvement is **parameter centralization**.

Before:

- important values were scattered across `main.py`, `fine_align.py`, `ai_detector.py`, and helper logic

Now:

- the main runtime tuning values are in `config.py`

That means the workflow is better:

## Old workflow

- find the number in some random file
- change it there
- hope nothing else overrides it

## New workflow

- open `config.py`
- tune the runtime value
- rerun

This is especially helpful for:

- survey confidence
- survey stability
- PD gains
- deadzone
- timeout
- settle frames
- strike settings

---

# 10. IMPORTANT CAUTIONS

## Do not change multiple subsystems at once

If you change:

- detector confidence
- survey filter
- PD gains
- strike settings

all in one pass, you will not know what helped and what broke things.

Change one group at a time.

## Tuner behavior is not the same as full survey behavior

A live tuning view may show detections, but the survey still may reject them later.

## Fine align depends on coarse move being reasonable

If triangulation is bad, PD will not save you.

## Keep laser tests conservative

Always test strike with low power / short duration first.

---

# 11. SHORT VERSION: WHERE TO LOOK FIRST

If you only remember one section, use this one.

## Need to tune survey?

Go to:

```python
AI_CONFIDENCE
SURVEY_BURST_COUNT
SURVEY_MIN_HITS
SURVEY_CLUSTER_RADIUS_PX
```

## Need to tune PD / fine align?

Go to:

```python
FINE_ALIGN_KP_X
FINE_ALIGN_KD_X
FINE_ALIGN_KP_Y
FINE_ALIGN_KD_Y
FINE_ALIGN_DEADZONE_PX
FINE_ALIGN_MAX_TIME_SEC
FINE_ALIGN_SETTLE_FRAMES
FINE_ALIGN_CROP_SCALE
```

## Need to tune strike?

Go to:

```python
LASER_FIRE_POWER
LASER_FIRE_DURATION_SEC
LASER_ARM_DELAY_SEC
```

## Need to fix geometry?

Go to:

```python
TRI_SIGN_X
TRI_SIGN_Y
TRI_X_GAIN
TRI_Y_GAIN
LASER_OFFSET_X_MM
LASER_OFFSET_Y_MM
CALIB_NPZ_PATH
RECT_NPZ_PATH
```

---

# 12. FINAL NOTE

This workspace is in a much better state now because the important knobs are no longer hidden all over the place.

The intended workflow is now:

1. diagnose the stage that failed
2. change the relevant group of config values
3. rerun
4. only open lower-level files if config tuning is not enough

That is the whole point of this rewrite.
