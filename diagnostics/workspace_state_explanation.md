# LaserWeeder_CleanRuntime workspace state explanation

Generated: 2026-05-02

Scope: read-only investigation, except for this report and the requested GrandMaster comparison patch files:

- `diagnostics/diff_cameras_grandmaster_vs_clean_current.patch`
- `diagnostics/diff_main_grandmaster_vs_clean_current.patch`

Important note: at the start of this investigation, `git status --short --branch` showed a clean tree. During the investigation, an already-running `bringup/06_survey_detect_only.py` process was visible briefly and then exited; after that, `bringup/logs/06_left.jpg` and `bringup/logs/06_right.jpg` appeared modified. I did not start that process and did not change code.

## 1. Current git/workspace state

Commands recorded at the start:

```text
pwd
/home/eli/LaserWeeder_CleanRuntime

readlink -f .
/home/eli/LaserWeeder_CleanRuntime

git status --short --branch
## clean-runtime-v2...origin/clean-runtime-v2

git log --oneline --decorate -10
2dbf162 (HEAD -> clean-runtime-v2, origin/clean-runtime-v2) finally FUCKING WORKING
607f991 Add safe mock gantry and runtime preflight
cdfa513 (origin/grandmaster-cleanup, valid-jetson-headless-controll, grandmaster-cleanup) post processing video with time stamps and status bar added
5bf3a0d post processing video with time stamps and status bar added
9b26725 Add render tool, TensorRT export, snap/qpoint fine-align, and PD manifest logging
5f02a64 Add cleaned GrandMaster laser weeder workspace

git rev-parse HEAD
2dbf162b0063327f11f5b4f714fe448058842e86

git remote -v
origin git@github.com:rekatihwile/SCI_Weeder.git (fetch)
origin git@github.com:rekatihwile/SCI_Weeder.git (push)

git branch --show-current
clean-runtime-v2
```

`git diff --stat` was empty at the start. The requested diff command against `main.py`, `hardware/cameras.py`, `config.py`, and `bringup/*.py` also produced no output at the start.

Interpretation:

- Branch: `clean-runtime-v2`.
- Commit: `2dbf162b0063327f11f5b4f714fe448058842e86`.
- Remote tracking: local `clean-runtime-v2` is aligned with `origin/clean-runtime-v2`.
- The likely pushed commit from yesterday is `2dbf162 finally FUCKING WORKING`.
- At investigation start, the working tree was clean.
- After this report/diff generation, the tree is no longer clean because this investigation created diagnostics files, and an external/current run modified `bringup/logs/06_left.jpg` and `bringup/logs/06_right.jpg`.
- This workspace is still separate from `/home/eli/LaserWeeder_GrandMaster`; it has its own path, branch, and working tree. Both are connected to the same GitHub repo, but they are separate local checkouts.

## 2. Runtime environment

The requested venv wrapper uses:

```text
python: /home/eli/venvs/laserweeder_cv412/bin/python
version: 3.8.10
cv2: 4.12.0
numpy: 1.23.5
PIL: 10.4.0
torch: 2.0.0+nv23.05 cuda: True
torchvision: 0.15.1
ultralytics: 8.4.26
pyserial: 3.5
```

This is the intended `/home/eli/venvs/laserweeder_cv412` environment. It is using the known-good OpenCV/Numpy pair for this workspace: `cv2 4.12.0` and `numpy 1.23.5`.

CUDA is available: `torch.cuda.is_available()` returned `True`.

Torchvision still shows a broken native extension warning:

```text
Failed to load image Python extension: ... torchvision/image.so: undefined symbol ...
```

That warning is not exactly the NMS crash, but it confirms the torchvision C++ extension installation is not healthy. Older diagnostics also show the direct NMS failure:

```text
Couldn't load custom C++ ops...
```

So the environment is usable with the patch, but torchvision native ops should not be trusted here.

## 3. NMS patch explanation

`bringup/_nms_patch.py` monkey-patches `torchvision.ops.nms` with a pure-PyTorch implementation. It sorts boxes by score, repeatedly keeps the highest-score box, computes IoU against the remaining boxes, and drops boxes above the IoU threshold. It then assigns that function to:

```python
torchvision.ops.nms = _nms_pytorch
```

Where it is imported:

- `main.py`, at the very top, before importing `vision.detectors.ai_detector` and therefore before Ultralytics/YOLO code.
- `bringup/02_yolo_detection.py`
- `bringup/06_survey_detect_only.py`
- `bringup/07_match_plan_only.py`
- `bringup/08_runtime_step_machine.py`

Other torchvision use:

- `vision/detectors/ai_detector.py` imports `from torchvision import models` for the qpoint model encoder.
- `ai_detector.py` imports Ultralytics YOLO, which can call torchvision NMS internally.

Is it needed? Yes, currently it looks needed for YOLO paths in this venv. The runtime imports show torchvision native extension trouble, and `diagnostics/eli_clean_camera_sequence_test.log` records the exact custom C++ ops crash during warmup without the patch.

Is it related to the camera issue? No. The NMS patch affects YOLO post-processing. The camera issue is about V4L2/OpenCV capture returning `ret=True` with `frame=None`, especially from the right camera. The patch does not open cameras, close cameras, reset USB, or change V4L2 settings.

What would happen if the patch were removed? AI/YOLO scripts would likely fail during warmup or inference with the torchvision custom C++ ops / NMS error. Camera-only script `01_camera_open.py` would not need it.

Recommendation: keep the NMS patch for now. It is a practical compatibility shim for this Jetson venv, and removing it would probably break YOLO bringup again.

## 4. Bringup ladder inventory

Files under `bringup`:

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
bringup/logs/...
```

Script summary:

| Script | Tests | Touches hardware | Opens cameras | Moves gantry | Uses YOLO | SSH-safe? |
|---|---|---:|---:|---:|---:|---|
| `00_env_check.py` | Python, packages, config, git HEAD | No | No | No | Imports Ultralytics only | Yes |
| `01_camera_open.py` | Stereo camera open/read, 30 pairs | Cameras | Yes | No | No | Usually yes, but it uses cameras |
| `02_yolo_detection.py` | YOLO warmup + one stereo pair detection | Cameras/GPU | Yes | No | Yes | Usually yes |
| `03_gantry_status.py` | GRBL serial status/settings read | Serial only | No | No intentional motion | No | Caution: opens serial and waits through GRBL reset grace |
| `04_gantry_home.py` | Homing | Gantry | No | Yes, homes | No | No, unless physically supervised |
| `05_gantry_move_survey.py` | Move to survey position | Gantry | No | Yes | No | No, unless physically supervised |
| `06_survey_detect_only.py` | Survey burst detection at current position | Cameras/GPU | Yes | No | Yes | Usually yes, if current gantry position is safe |
| `07_match_plan_only.py` | Survey detection, matching, triangulation, planning | Cameras/GPU | Yes | Mock only | Yes | Usually yes |
| `08_runtime_step_machine.py` | Minimal pipeline through plan | Cameras/GPU, mock gantry by default | Yes | Mock by default | Yes | Yes only while `USE_REAL_GANTRY=False` |

Latest known bringup log results:

- `00_env_check.log`: PASS. It recorded HEAD `607f991`, so it predates the current `2dbf162` commit.
- `01_camera_open.log`: PASS, 30/30 pairs, before retry-fix logging.
- `01_after_copy_grandmaster.log`: FAIL, 0/30 pairs. It showed the right camera returning `ret=True` but `R_None=True` while properties still said `1280x720 @ 30 MJPG`.
- `01_camera_open_retryfix_1.log`: PASS after two failed open/drain attempts and success on the third.
- `01_camera_open_retryfix_2.log`: PASS on first attempt.
- `01_camera_open_retryfix_3.log`: PASS on first attempt.
- `02_yolo_detection.log`: PASS, YOLO detections on both cameras.
- `02_after_copy_grandmaster.log`: PASS, YOLO detections on both cameras.
- `02_yolo_detection_retryfix.log`: PASS, first camera-open attempt failed validation, second succeeded.
- `03_gantry_status.log`: PASS, serial status response received.
- `04_gantry_home.py`: no log found.
- `05_gantry_move_survey.py`: no real run log found; only an empty `05c_grbl_settings_debug.logcd`.
- `06_survey_detect_only.py`: no `.log` file found, but `06_left.jpg` and `06_right.jpg` exist and were updated while this investigation was running.
- `07_match_plan_only.py`: no log found.
- `08_runtime_step_machine.py`: no log found.

## 5. Current camera lifecycle in code

In `main.py`:

- `StereoCameras()` is created in `INIT`.
- `AIDetector()` is also created in `INIT`.
- In `HOME`, YOLO warmup runs before `cameras.open()`.
- Then `cameras.open()` starts cameras and, if `RECORD_TRIAL=True`, auto-starts recording.
- Before survey detection, `_flush_camera_buffer(cameras, n=8)` drains frames.
- Main always enters `finally`, saves a manifest, calls `cameras.stop_recording()`, then `cameras.close()`, then `gantry.close()`.

In `hardware/cameras.py`:

- `open()` opens `/dev/video0` and `/dev/video2` through OpenCV/V4L2.
- It explicitly sets MJPG before width/height/FPS.
- It applies V4L2 camera settings.
- It drains up to 150 iterations, about 5 seconds, until both cameras return `ret=True`, non-`None` frames, and shape length 3.
- It retries the full open/release cycle up to 3 times. If all attempts fail, it raises instead of continuing with invalid cameras.
- After validation, it starts a background grab thread.
- If `RECORD_TRIAL` and `start_recorder=True`, it starts raw recording.

Read behavior:

- `read_pair()` uses a camera lock, calls `grab()` on both cameras, then `retrieve()` on both.
- It accepts only `ret_l and ret_r and frame_l is not None and frame_r is not None`.
- On failure, it logs `ret_l`, `ret_r`, `L_None`, `R_None`, shapes, and capture properties for the first 10 failures.
- It retries 5 times before raising `Failed to read stereo pair after multiple retries.`

Close behavior:

- `close()` stops the background grab thread first.
- It joins the background thread with a 2-second timeout.
- It calls `stop_recording()`.
- It releases both `VideoCapture` objects if present.

Current limitations:

- `close()` does not set `self.left` and `self.right` to `None` after release.
- `close()` does not sleep after release.
- `close()` does not call `cv2.destroyAllWindows()`.
- If the background thread fails to join within 2 seconds, `close()` does not log a warning.
- The background thread is supposed to stop before release. Because it is daemonized and exceptions are swallowed inside the loop, the chance of it reading after close is low in normal flow, but not impossible if join times out.
- `stop_recording()` stores stats after calling `release()`, but because it asks `get_recording_stats()` before setting `_recorder = None`, stats should still be available.

Is there a retry/reopen if the right frame is `None`? During `open()`, yes. During later `read_pair()` calls, no. Later read failures retry reads but do not close/reopen the devices.

## 6. Compare camera lifecycle to GrandMaster

Requested diff files were generated:

- `diagnostics/diff_cameras_grandmaster_vs_clean_current.patch`
- `diagnostics/diff_main_grandmaster_vs_clean_current.patch`

Camera behavior inherited from GrandMaster:

- V4L2/OpenCV camera opening.
- MJPG-before-resolution negotiation.
- V4L2 startup setting application.
- Post-open drain loop.
- Background grab thread.
- Raw/live recorder structure.
- `read_pair()` grab/retrieve pattern.
- `close()` stops background thread, stops recorder, releases both captures.

Different in CleanRuntime:

- CleanRuntime added a 3-attempt open validation/retry loop.
- CleanRuntime refuses to proceed after invalid drain attempts; GrandMaster warned and continued.
- CleanRuntime validates frame shape, not just non-`None`.
- CleanRuntime added `RECORD_VIDEO_OVERLAY` handling.
- CleanRuntime `main.py` is a much smaller one-cycle state machine; GrandMaster has a much larger runtime with cycle/config/render extras.
- GrandMaster `main.py` warmed YOLO after `cameras.open()` on the first cycle. CleanRuntime currently warms YOLO before `cameras.open()`.

Is CleanRuntime now using the same method that worked? Mostly, plus stricter validation. The core open settings and drain pattern came from GrandMaster, but CleanRuntime adds a useful retry/reopen wrapper. The logs show that this retry wrapper helped: several runs failed validation on attempt 1 and then succeeded on attempt 2 or 3.

Remaining differences that could explain "works once, fails next run":

- `close()` still does not set captures to `None`, sleep after release, or destroy OpenCV windows.
- There is no reusable "full camera recover" function for mid-run `read_pair()` failures.
- Warmup order differs from GrandMaster. CleanRuntime warms YOLO before opening cameras. The code comments say Jetson can return `ret=True, frame=None` after GPU warmup until USB stabilizes; CleanRuntime mitigates this with post-open drain/retry.
- If the kernel/UVC device itself gets into a bad state, process-level release may not be enough; the retry loop helps but cannot guarantee USB device recovery.

## 7. Recent trial logs and failure pattern

Recent trial directories:

```text
trial_recordings/trial_001_20260501_205325
```

Recent metrics JSON:

```text
20260501_205325.json user_aborted det=(19,21) matches=18 planned=18 attempted=7 fired=3 frames=1936 time=129.713
20260501_203303.json failed       det=(None,None) frames=35 time=2.478
20260501_203144.json failed       det=(None,None) frames=33 time=2.406
20260501_203044.json failed       det=(None,None) frames=16 time=1.11
20260501_202936.json failed       det=(None,None) frames=12 time=0.839
20260501_202811.json failed       det=(None,None) frames=12 time=0.858
20260501_200857.json failed       det=(None,None) frames=0 time=0.807
20260501_200745.json failed       det=(None,None) frames=0 time=0.82
20260501_200610.json complete     det=(16,0) matches=0 planned=0 frames=25
20260501_200451.json complete     det=(0,3) matches=0 planned=0 frames=24
20260501_200228.json complete     det=(3,0) matches=0 planned=0 frames=25
20260501_195639.json failed       det=(None,None) frames=0
```

Which succeeded:

- Camera bringup succeeded repeatedly after the retry fix.
- YOLO detection bringup succeeded.
- Main/runtime had complete survey-only style runs around `20:02`, `20:04`, and `20:06`.
- The latest full-ish trial at `20:53:25` reached 19 left detections, 21 right detections, 18 matches, 18 planned targets, attempted 7 targets, fired 3 simulated/allowed strikes, and ended as `user_aborted`.

Which failed:

- `01_after_copy_grandmaster.log` failed with 0/30 valid pairs because the right camera returned `ret=True` with `frame=None`.
- Several main runs between `20:07` and `20:33` failed early. The later ones with frame counts failed with `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`, not a camera read error in the visible logs.

Pattern:

- Yes, the logs fit a "works once, then next open can be invalid" pattern: `01_camera_open_retryfix_1` needed three open attempts, and `02_yolo_detection_retryfix` needed two.
- The exact camera failure captured was right camera `ret=True`, `R_None=True`, while both cameras still claimed MJPG/1280x720/30.
- After failed main runs, logs show raw recorder shutdown and GRBL close/reset messages. That suggests normal `finally` cleanup executed for those runs.
- The latest `trial_001_20260501_205325/recording_meta.json` says status complete, 1937 frame pairs saved, 0 dropped.
- I did not find evidence in the logs that a recorder thread was left alive. Live process checks at the end did not show a running `main.py` or bringup Python process.

## 8. Current device/process state

Device nodes exist:

```text
/dev/video0
/dev/video1
/dev/video2
/dev/video3
```

V4L2 devices:

```text
USB Cam: USB Cam (usb-3610000.xhci-4.3): /dev/video0 /dev/video1
USB Cam: USB Cam (usb-3610000.xhci-4.4): /dev/video2 /dev/video3
```

`/dev/video0 --all` and `/dev/video2 --all` both currently report default/current format as `640x480 YUYV @ 30`. That is not alarming by itself; it is just the current/default V4L2 state when OpenCV is not actively holding them in MJPG mode.

Format support was confirmed with `--list-formats-ext`:

- `/dev/video0` supports MJPG `1280x720` at 30 fps.
- `/dev/video2` supports MJPG `1280x720` at 30 fps.
- YUYV `1280x720` is only 10 fps, which matches the code comment that MJPG is important.

Process ownership:

- `fuser -v /dev/video0 /dev/video2` returned no holders.
- `lsof /dev/video0 /dev/video2` returned no holders.
- A `bringup/06_survey_detect_only.py` process was briefly visible during investigation, but a later `ps` check showed no `main.py`/bringup Python process alive.
- Remaining Python-like processes were system/VS Code/Pylance helpers, not LaserWeeder runtime processes.

## 9. Camera recovery options over SSH

Do not execute these without explicit approval. Safest to most aggressive:

A. Python-level recovery:

- Stop recorder.
- Stop background grab thread.
- Release left/right `VideoCapture`.
- Sleep briefly.
- Reopen cameras.
- Reapply MJPG/1280x720/30 and V4L2 camera settings.
- Validate 10 consecutive stereo pairs before returning success.

B. Shell-level process check:

- Confirm no user Python process holds `/dev/video*`.
- Use `fuser`/`lsof` to identify holders.
- Restart only the relevant user Python process if needed.

C. V4L2 settings reset:

- Reapply camera settings with `v4l2-ctl`.
- Force MJPG/1280x720/30 through the capture code or a controlled test.

D. USB/device reset:

- Identify the USB bus/device for each camera.
- Use `usbreset` if installed and explicitly authorized.
- Or unbind/rebind the known USB device path.
- Risk: resets the physical USB camera device, may disturb both cameras or other devices on the hub, and can leave device numbering changed if done carelessly.

E. Full system reboot:

- `sudo reboot`.
- Most aggressive but often clears UVC/USB/kernel camera state.

Recommended future code improvement: add a `StereoCameras.recover()` or `reopen_and_validate()` method that performs level A cleanly and logs each step. Use it in two places: after failed open validation, and optionally after mid-run `read_pair()` failures. Keep it camera-only; do not touch GRBL or laser state.

## 10. Final summary for Eli

1. What is currently working?

The venv is correct, CUDA is available, camera-only bringup passes after the retry fix, YOLO bringup passes with the NMS patch, and the latest full runtime trial reached survey, matching, planning, fine align attempts, and 3 fired/simulated strike events before user abort.

2. What probably caused the "works once then fails" camera problem?

The right UVC camera sometimes opens in a state where OpenCV reports `ret=True` but returns `frame=None`. The camera properties still look correct, so this is probably a USB/UVC/OpenCV stream readiness problem, not a config typo. The new open-drain-retry logic is directly aimed at this and appears to help.

3. Is NMS patch relevant or unrelated?

Unrelated to the camera issue. It is relevant to YOLO because torchvision native ops are broken in this venv.

4. Is the venv currently correct?

Yes: `/home/eli/venvs/laserweeder_cv412/bin/python`, `cv2 4.12.0`, `numpy 1.23.5`, CUDA available.

5. Is the camera close/release path robust enough?

Better than before, but not fully robust. It stops the background thread, stops recording, and releases both cameras. It does not set captures to `None`, sleep after release, call `cv2.destroyAllWindows()`, warn if the background thread fails to stop, or provide a mid-run recover/reopen path.

6. What is the smallest next code improvement?

Add a camera-only recovery/reopen method: stop recorder, stop background thread, release both captures, set them to `None`, sleep, reopen, and require 10 valid stereo pairs.

7. Should we continue with the bringup ladder or `main.py`?

Continue the bringup ladder first. Run camera and YOLO checks before `main.py`, especially over SSH.

8. What should Eli run first next time over SSH before `main.py`?

Start with:

```bash
./run_with_eli_venv.sh bringup/01_camera_open.py | tee bringup/logs/01_camera_open.log
```

If that passes, run:

```bash
./run_with_eli_venv.sh bringup/02_yolo_detection.py | tee bringup/logs/02_yolo_detection.log
```

Only then consider the higher ladder or `main.py`.
