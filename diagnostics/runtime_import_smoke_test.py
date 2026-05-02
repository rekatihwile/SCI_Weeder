"""
diagnostics/runtime_import_smoke_test.py
-----------------------------------------
Safe import smoke test for the LaserWeeder clean runtime.

Rules:
- Imports every key module to verify the import graph is intact.
- Does NOT instantiate hardware classes (Gantry, StereoCameras, AIDetector).
- Does NOT open serial ports, cameras, or GPU models.
- Does NOT move anything.
"""

import sys
import os
import traceback

# Ensure the repo root is on sys.path when run as a script.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_passed = []
_failed = []


def _try(label, fn):
    try:
        fn()
        _passed.append(label)
        print(f"  [OK]   {label}")
    except Exception as e:
        _failed.append((label, e))
        print(f"  [FAIL] {label}")
        traceback.print_exc()


print()
print("=" * 62)
print("  RUNTIME IMPORT SMOKE TEST")
print("=" * 62)

# --- config ---
_try("import config", lambda: __import__("config"))

# --- hardware (class reference only, no instantiation) ---
_try(
    "import hardware.cameras.StereoCameras (no instantiation)",
    lambda: __import__("hardware.cameras", fromlist=["StereoCameras"]),
)

_try(
    "import hardware.gantry.Gantry (no instantiation)",
    lambda: __import__("hardware.gantry", fromlist=["Gantry"]),
)

# --- vision ---
_try(
    "import vision.detectors.ai_detector.AIDetector (no instantiation)",
    lambda: __import__("vision.detectors.ai_detector", fromlist=["AIDetector"]),
)

# --- control ---
_try(
    "import control.coarse_move.TriangulationCoarseMover",
    lambda: __import__("control.coarse_move", fromlist=["TriangulationCoarseMover"]),
)

# --- vision.matching ---
_try(
    "import vision.matching.match_points",
    lambda: __import__("vision.matching", fromlist=["match_points"]),
)

# --- planning ---
_try(
    "import planning.target_planner.plan_targets",
    lambda: __import__("planning.target_planner", fromlist=["plan_targets"]),
)

# --- pipeline.preflight ---
_try(
    "import pipeline.preflight.print_preflight",
    lambda: __import__("pipeline.preflight", fromlist=["print_preflight"]),
)

# --- Summary ---
print()
print("=" * 62)
print(f"  Results: {len(_passed)} passed, {len(_failed)} failed")
print("=" * 62)

if _failed:
    print("\n  FAILURES:")
    for label, exc in _failed:
        print(f"    - {label}: {exc}")
    print()

# --- Run preflight if all imports passed ---
if not _failed:
    print()
    from pipeline.preflight import print_preflight
    print_preflight()
else:
    print("[SMOKE TEST] Skipping preflight — fix import failures first.")
    sys.exit(1)
