# Runtime Refactor Plan
**Branch:** clean-runtime-v2  
**Date:** 2026-05-01  
**Status:** Plan only — no code has been moved yet.

---

## 1. What is main.py currently responsible for?

`main.py` is a monolithic driver that owns the following concerns simultaneously:

| Concern | How it manifests in main.py |
|---|---|
| **State machine** | `while state != "DONE"` with 9 named string states (INIT, HOME, SURVEY, SURVEY_CONFIRM, DETECT, MATCH, PLAN, EXECUTE, DONE) |
| **Hardware initialisation** | Gantry and StereoCameras construction, camera open, detector build, coarse mover creation |
| **Operator configuration resolution** | `_resolve_burst_count()`, `_resolve_point_mode()` — translate config flags into per-call values |
| **Camera buffer management** | `_flush_camera_buffer()` |
| **Manifest construction** | `manifest` dict is assembled inline across all states; `_save_manifest()` writes it |
| **Recording context updates** | `_update_recording_context()` called before/after every hardware action to annotate the camera recorder |
| **Metrics instrumentation** | `_attach_recording_metrics()`, `_add_run_total()`, `_save_metrics()`, `_metrics_snapshot()` — logger start/end wrapped around every section and target |
| **Target data compaction** | `_compact_target_list()`, `_compact_hits()`, `_gantry_xy()`, `_active_target_xy()` — helpers that format pipeline state for recording |
| **Survey → detect → match orchestration** | Calls `coarse_mover.detect_stable_points()`, `match_points()`, user confirmation prompt |
| **Triangulation + planning orchestration** | Calls `coarse_mover.fit_epipolar()`, `coarse_mover.solve_all_from_pose()`, `plan_targets()`, saves workspace targets, optionally shows plots |
| **Per-target execution loop** | Bounds check, duplicate check, travel, TRIANGULATION_ONLY_MODE branch, fine-align, fire, per-target manifest entry, per-target metrics |
| **Error handling and teardown** | KeyboardInterrupt and generic exception handling, final manifest save, camera/gantry close |
| **Debug visualisation gating** | Calls to `show_workspace_triangulation_map`, `show_match_debug_view` guarded by `HAS_DISPLAY` / config flags |

**Line count:** 672 lines. The `main()` function alone spans approximately 430 lines.

---

## 2. Which responsibilities should stay in main.py?

After refactoring, main.py should remain the **entry point and composition root** only:

- **Import and wire** all pipeline modules together.
- **Instantiate** hardware objects (gantry, cameras, detector, coarse mover) in one place.
- **Start the state machine** by calling `pipeline.state_machine.run(ctx)`.
- **Top-level exception boundary** — catch KeyboardInterrupt and unhandled Exception, call teardown.
- **Teardown** — `cameras.close()`, `gantry.close()`.

main.py should become approximately 60–80 lines.  It should contain no business logic.

---

## 3. Which responsibilities should move into modules?

| Responsibility | Target module |
|---|---|
| State machine loop and state transitions | `pipeline/state_machine.py` |
| HOME and camera/detector startup | `pipeline/survey.py` (setup phase) |
| DETECT: flush, detect_stable_points, timing | `pipeline/survey.py` |
| MATCH: match_points, user confirm | `pipeline/matching_pipeline.py` |
| PLAN: fit_epipolar, solve, plan_targets, save, plots | `pipeline/planning_pipeline.py` |
| EXECUTE loop: per-target travel, fine-align, fire | `pipeline/execution.py` |
| `_update_recording_context()` and all its helpers | `pipeline/recording_context.py` |
| `_attach_recording_metrics()`, `_add_run_total()`, `_metrics_snapshot()`, `_save_metrics()` | `pipeline/metrics_hooks.py` |
| `_save_manifest()`, `_compact_target_list()`, `_compact_hits()` | `pipeline/manifest.py` |
| `_resolve_burst_count()`, `_resolve_point_mode()` | `pipeline/survey.py` (or a small `pipeline/cv_config.py`) |
| Rectification and calibration loading from coarse_move | `vision/rectification.py` |
| Remote test for rectification | `dev_tools/remote/rectify_test.py` |

---

## 4. What exact new files should exist?

```
pipeline/
    __init__.py               (empty, already exists)
    preflight.py              (already created)
    state_machine.py          NEW — RunContext dataclass + run() state loop
    survey.py                 NEW — open_cameras(), run_detect(), _flush_camera_buffer(),
                                    _resolve_burst_count(), _resolve_point_mode()
    matching_pipeline.py      NEW — run_match(), user confirm logic
    planning_pipeline.py      NEW — run_plan(), fit+solve+plan_targets, save, plot gating
    execution.py              NEW — run_execute(), per-target loop, TRIANGULATION_ONLY branch
    recording_context.py      NEW — update_recording_context(), _gantry_xy(),
                                    _compact_target_list(), _compact_hits(),
                                    _active_target_xy(), _metrics_snapshot()
    metrics_hooks.py          NEW — attach_recording_metrics(), add_run_total(),
                                    save_metrics(), build_logger()
    manifest.py               NEW — save_manifest(), compact_target_entry(),
                                    compact_hit_entry()

vision/
    rectification.py          NEW — load_calibration(), undistort_points(),
                                    triangulate_point_rectified(), fit_epipolar()
                                    (extracted from control/coarse_move.py)

dev_tools/
    remote/
        rectify_test.py       NEW — offline test for vision/rectification.py
                                    using saved calibration files
```

---

## 5. What functions should move first?

Move in this exact order to minimise breakage at each step.  Each step must
pass the compile check and smoke test before the next step starts.

### Round 1 — Zero-risk extractions (pure functions, no imports changed)

1. `_save_manifest()`, `_compact_target_list()`, `_compact_hits()` →
   **`pipeline/manifest.py`**  
   These are pure data-formatting functions with no hardware dependency.

2. `_metrics_snapshot()`, `_attach_recording_metrics()`, `_add_run_total()`,
   `_save_metrics()` →  
   **`pipeline/metrics_hooks.py`**  
   Depend only on the logger interface and cameras.get_recording_stats().

3. `_gantry_xy()`, `_active_target_xy()`, `_update_recording_context()`,
   `_compact_target_list()`, `_compact_hits()` →  
   **`pipeline/recording_context.py`**  
   Depends on cameras.set_recording_context() only.

### Round 2 — CV config helpers

4. `_resolve_burst_count()`, `_resolve_point_mode()`, `_flush_camera_buffer()` →
   **`pipeline/survey.py`**  
   No hardware instantiation; tests trivially in isolation.

### Round 3 — State handlers (one at a time)

5. DETECT handler body → `pipeline/survey.py::run_detect(ctx)`
6. MATCH handler body → `pipeline/matching_pipeline.py::run_match(ctx)`
7. PLAN handler body → `pipeline/planning_pipeline.py::run_plan(ctx)`
8. EXECUTE handler body → `pipeline/execution.py::run_execute(ctx)`

### Round 4 — State machine

9. `while state != "DONE":` loop → `pipeline/state_machine.py::run(ctx)`
10. INIT and HOME handler bodies → `pipeline/state_machine.py::_state_init(ctx)` /
    `pipeline/survey.py::run_home(ctx)`

### Round 5 — Rectification (later, separate PR)

11. `_triangulate_point_rectified()`, `_normalize_rectified_calibration_units_to_meters()`,
    calibration loading block in `TriangulationCoarseMover.__init__()` →  
    **`vision/rectification.py`**

---

## 6. What should not be touched yet?

- `control/coarse_move.py` — TriangulationCoarseMover is large (657 lines) and mixes
  rectification, survey detection, stereo solve, and motion. It must be extracted
  only after vision/rectification.py exists (Round 5). Do not touch in Rounds 1–4.

- `control/fine_align.py` — 1213 lines of carefully tuned closed-loop tracking.
  Do not refactor until the state machine extraction is complete and all end-to-end
  probe tests pass. Any change here risks the PD loop.

- `hardware/cameras.py` — 1214 lines; has its own recording infrastructure.
  No changes needed; it already has a clean API boundary.

- `hardware/gantry.py` — working and stable. No changes needed.

- `vision/matching.py` — pure-function stereo matching. No changes needed until
  rectification is extracted, and even then matching.py is untouched.

- `config.py` — only add new flags, never restructure existing ones.

- `params/` — calibration files must never be moved or renamed.

- Any existing probe/diagnostic scripts — must continue to pass unchanged.

---

## 7. What tests/probes must pass after each extraction?

Every extraction round must pass **all three gates** before the next round begins:

### Gate A — Compile check (after every file change)
```
./run_with_eli_venv.sh -m py_compile \
    main.py config.py \
    pipeline/manifest.py pipeline/metrics_hooks.py \
    pipeline/recording_context.py pipeline/survey.py \
    pipeline/matching_pipeline.py pipeline/planning_pipeline.py \
    pipeline/execution.py pipeline/state_machine.py
```

### Gate B — Import smoke test (after every round)
`diagnostics/runtime_import_smoke_test.py` must still show 8/8 OK.

### Gate C — Camera sequence probe (after Rounds 3 and 4)
`diagnostics/main_camera_sequence_probe.py` must still show:
```
Part 3  no-YOLO   flush=8/8   burst=30/30
Part 4  with-YOLO flush=8/8   burst=30/30
```

### Gate D — Mock gantry smoke test (after Round 4)
`diagnostics/mock_gantry_smoke_test.py` must still show 24/24 passed.

### Gate E — New module unit tests (per round)

| Round | New test |
|---|---|
| Round 1 (manifest) | `diagnostics/test_manifest_helpers.py` — assert compact_target_list and compact_hits produce correct dicts from sample input |
| Round 2 (metrics) | `diagnostics/test_metrics_hooks.py` — assert save_metrics handles None logger gracefully |
| Round 3 (survey) | `diagnostics/test_cv_config.py` — assert resolve_burst_count / resolve_point_mode edge cases |
| Round 4 (state handlers) | `diagnostics/test_pipeline_states.py` — run each state function with MockGantry + mock camera fixture |
| Round 5 (rectification) | `dev_tools/remote/rectify_test.py` — load saved npz files, triangulate a known point, assert within tolerance |

---

## 8. Where should rectification fit later?

`control/coarse_move.py` currently owns three separate concerns:

1. **Calibration loading** — npz files loaded in `__init__`, fisheye parameters stored as instance attributes.
2. **Rectification / triangulation** — `_triangulate_point_rectified()`, `_normalize_rectified_calibration_units_to_meters()`, `_unflip_point_180()`, `fit_epipolar()`.
3. **Survey detection** — `detect_stable_points()`, `_cluster_burst_points()`, `_scale_stable_to_calib()`.
4. **Motion** — `move_to_absolute_target()`.

Proposed future split:

```
vision/rectification.py
    load_stereo_calibration(npz_path, rect_path)  → CalibData namedtuple
    undistort_points(pts, K, D, R, P)
    triangulate_point_rectified(uL, vL, uR, vR, calib)
    fit_epipolar(matched_targets, calib)
    solve_target_xyz(target, calib, ref_x, ref_y)
```

`TriangulationCoarseMover` would then import `CalibData` from `vision.rectification`
and delegate all rectification calls to it.  The class itself becomes a
survey-and-motion coordinator with no low-level camera math inside.

**This extraction must happen after Rounds 1–4** because it requires modifying
`TriangulationCoarseMover.__init__` and all callers of the calibration attributes.
It is gated behind a passing `dev_tools/remote/rectify_test.py` that validates
the triangulation output against known reference coordinates before any other
code is changed.

---

## 9. What should the final main.py state machine look like?

```python
# main.py — target state after all refactor rounds complete

from pipeline.state_machine import RunContext, run
from pipeline.preflight import print_preflight


def main():
    print_preflight()
    ctx = RunContext.from_config()
    try:
        run(ctx)
    except KeyboardInterrupt:
        ctx.teardown(status="user_aborted")
    except Exception as e:
        print(f"\nERROR: {e}")
        ctx.teardown(status="failed")
    finally:
        ctx.teardown()


if __name__ == "__main__":
    main()
```

`RunContext` is a dataclass (or plain class) that holds all pipeline state that
currently lives as locals in `main()`:

```python
@dataclass
class RunContext:
    # Hardware
    gantry: object          # Gantry | MockGantry
    cameras: StereoCameras
    detector: object        # AIDetector | ManualDetectorLocal
    coarse_mover: TriangulationCoarseMover
    logger: object | None

    # Trial state
    state: str = "INIT"
    trial_timestamp: str = field(default_factory=...)
    manifest: dict = field(default_factory=...)
    target_queue: list = field(default_factory=list)
    actual_hits: list = field(default_factory=list)
    left_detections: list = field(default_factory=list)
    right_detections: list = field(default_factory=list)
    matched_targets: list = field(default_factory=list)

    @classmethod
    def from_config(cls) -> "RunContext":
        # Reads config, constructs hardware objects, returns a ready context.
        ...

    def teardown(self, status: str = "complete"):
        # Saves manifest, stops recording, closes cameras/gantry.
        ...
```

`pipeline/state_machine.py::run(ctx)` is:

```python
def run(ctx: RunContext):
    while ctx.state != "DONE":
        if ctx.state == "INIT":
            _state_init(ctx)
        elif ctx.state == "HOME":
            survey.run_home(ctx)
        elif ctx.state in ("SURVEY", "SURVEY_CONFIRM", "DETECT"):
            survey.run_detect(ctx)
        elif ctx.state == "MATCH":
            matching_pipeline.run_match(ctx)
        elif ctx.state == "PLAN":
            planning_pipeline.run_plan(ctx)
        elif ctx.state == "EXECUTE":
            execution.run_execute(ctx)
```

Each `run_X(ctx)` function reads from `ctx`, mutates `ctx.state` to advance
the machine, and returns.  No global variables.  No cross-module coupling
except through `ctx`.

---

## Summary of proposed module boundaries

```
main.py                       Entry point, composition root, top-level teardown
pipeline/state_machine.py     RunContext, run(), _state_init(), state dispatch
pipeline/survey.py            run_home(), run_detect(), _flush_camera_buffer(),
                              _resolve_burst_count(), _resolve_point_mode()
pipeline/matching_pipeline.py run_match(), user-confirm prompt
pipeline/planning_pipeline.py run_plan(), fit+solve+plan_targets, plot gating
pipeline/execution.py         run_execute(), per-target loop, tri-only branch
pipeline/recording_context.py update_recording_context(), compact helpers
pipeline/metrics_hooks.py     Logger wrappers, section/target timing helpers
pipeline/manifest.py          save_manifest(), compact_target_list(),
                              compact_hits(), compact_target_entry()
vision/rectification.py       CalibData, load_stereo_calibration(),
                              triangulate_point_rectified(), fit_epipolar()
dev_tools/remote/rectify_test.py  Offline calibration accuracy test
```

None of these files introduce new behaviour.  Every extraction is a pure
relocation of existing logic with identical semantics.  The first observable
change in runtime behaviour will only come from subsequent feature work built
on top of this structure.
