"""
bringup/08_runtime_step_machine.py
------------------------------------
Run the shared runtime orchestrator through survey, match, and plan.

USE_REAL_GANTRY = False  =>  uses MockGantry (no serial, no motion)

Does NOT:
  - fire laser
  - fine-align
  - execute target moves
  - record trial video

Run with:
    ./run_with_eli_venv.sh bringup/08_runtime_step_machine.py | tee bringup/logs/08_runtime_step_machine.log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runtime import run_runtime


USE_REAL_GANTRY = False


def main():
    run_runtime(use_real_gantry=USE_REAL_GANTRY, execute_targets=False)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
