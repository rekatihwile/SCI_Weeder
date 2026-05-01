"""
Terminal UI for the Laser Weeder.

Design goals:
  - Works over SSH (pure ANSI, no curses)
  - ANSI color/bold only when stdout is a real TTY; plain text otherwise
  - Fine-align status updates in-place with \\r so the terminal doesn't flood
  - Compact, readable layout; no 90-char walls of asterisks
"""

import os
import sys
import time
import shutil

# ── ANSI helpers ─────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def _bold(t):   return _c("1",    t)
def _cyan(t):   return _c("36",   t)
def _green(t):  return _c("32",   t)
def _yellow(t): return _c("33",   t)
def _red(t):    return _c("31",   t)
def _gray(t):   return _c("90",   t)
def _white(t):  return _c("97",   t)


def _w() -> int:
    return shutil.get_terminal_size((90, 24)).columns


def _rule(char="─", n=None):
    return char * (n or _w())


def _header(title: str):
    w = _w()
    bar = _rule("═", w)
    print(_bold(_cyan(bar)))
    print(_bold(_cyan(f"  {title}")))
    print(_bold(_cyan(bar)))


def _sep():
    print(_gray(_rule("─")))


# ── fine-align in-place state ────────────────────────────────────────────────

_fine_in_place  = False   # True while we're mid in-place line
_last_fine_time = 0.0


def _finalize_fine_line():
    """Print a newline to close an in-place fine-align line if one is open."""
    global _fine_in_place
    if _fine_in_place and _IS_TTY:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _fine_in_place = False


# ── public API ────────────────────────────────────────────────────────────────

def clear_screen():
    _finalize_fine_line()
    if _IS_TTY:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    else:
        print("\n" + _rule("─"))


def clear_current_target_line():
    _finalize_fine_line()


def print_header(title: str):
    _finalize_fine_line()
    _header(title)


def print_section(title: str):
    _finalize_fine_line()
    _sep()
    print(_bold(title))
    _sep()


def print_kv(label: str, value, indent: int = 0):
    pad = " " * indent
    label_str = _gray(f"{label:<28}")
    print(f"{pad}{label_str} {value}")


def print_status_banner(state: str, subtitle=None):
    clear_screen()
    _header(f"LASER WEEDER  ·  {state}")
    if subtitle:
        print_kv("Info", subtitle)


# ── survey & match ────────────────────────────────────────────────────────────

def print_global_survey_ready(x: float, y: float):
    clear_screen()
    _header("LASER WEEDER  ·  GLOBAL SURVEY")
    print_kv("Survey position", f"X={x:.1f} mm   Y={y:.1f} mm")
    print()
    print(f"  {_bold('Enter')} = run survey    {_bold('q')} = quit")
    print()


def print_global_survey_results(left_count: int, right_count: int, matched_count: int):
    _sep()
    print(_bold("  Survey results"))
    ok = _green if matched_count > 0 else _red
    print_kv("  Stable left",    left_count)
    print_kv("  Stable right",   right_count)
    print_kv("  Matched targets", ok(str(matched_count)))
    print()


def print_match_summary(matched_targets, unmatched_left, unmatched_right):
    _finalize_fine_line()
    w = _w()
    print()
    _sep()
    print(_bold("  Match summary"))
    _sep()
    print_kv("  Matched",         _green(str(len(matched_targets))))
    print_kv("  Unmatched left",  len(unmatched_left))
    print_kv("  Unmatched right", len(unmatched_right))

    if matched_targets:
        print()
        hdr = f"  {'#':>3}  {'Left (x,y)':<18}  {'Right (x,y)':<18}  {'yD':>5}  {'disp':>5}  Score"
        print(_gray(hdr))
        _sep()
        for i, t in enumerate(matched_targets):
            lp  = str(t["left_px"])
            rp  = str(t["right_px"])
            sc  = t["score"]
            yd  = float(t.get("y_diff_px", abs(t["left_px"][1] - t["right_px"][1])))
            disp = float(t.get("disp_px", abs(t["left_px"][0] - t["right_px"][0])))
            col = _green if sc >= 0.7 else (_yellow if sc >= 0.4 else _red)
            yd_col = _green if yd <= 15 else (_yellow if yd <= 25 else _red)
            print(f"  {i:>3}  {lp:<18}  {rp:<18}  {yd_col(f'{yd:5.1f}')}  {disp:5.1f}  {col(f'{sc:.3f}')}")

    if unmatched_left:
        print(_yellow(f"\n  Unmatched LEFT: ") + ", ".join(str(p) for p in unmatched_left))
    if unmatched_right:
        print(_yellow(f"  Unmatched RIGHT: ") + ", ".join(str(p) for p in unmatched_right))
    print()


# ── target execution ──────────────────────────────────────────────────────────

def print_workspace_plan(absolute_targets):
    _finalize_fine_line()
    print()
    _sep()
    print(_bold("  Workspace plan"))
    _sep()
    if not absolute_targets:
        print("  No targets planned.")
        return
    print(_gray(f"  {'#':>3}  {'X (mm)':>10}  {'Y (mm)':>10}"))
    for i, s in enumerate(absolute_targets, 1):
        tx, ty = s["target_xy_mm"]
        print(f"  {i:>3}  {tx:>10.2f}  {ty:>10.2f}")
    print()


def show_current_target(i: int, total: int, solved_target):
    _finalize_fine_line()
    clear_screen()
    tx, ty = solved_target["target_xy_mm"]
    frac   = i / max(total, 1)
    bar_w  = 24
    filled = int(frac * bar_w)
    bar    = _green("█" * filled) + _gray("░" * (bar_w - filled))

    _header(f"TARGET  {i} / {total}  [{bar}]")
    print_kv("  Coarse X", f"{tx:.2f} mm")
    print_kv("  Coarse Y", f"{ty:.2f} mm")

    if "source_target" in solved_target:
        src = solved_target["source_target"]
        if "score" in src:
            sc = src["score"]
            col = _green if sc >= 0.7 else _yellow
            print_kv("  Match score", col(f"{sc:.3f}"))
    print()


def print_target_result(i: int, total: int, solved_target, actual_entry=None):
    _finalize_fine_line()
    tx, ty = solved_target["target_xy_mm"]
    print(_green(f"  ✓ Target {i}/{total} locked"))
    print_kv("    Triangulated", f"({tx:.2f}, {ty:.2f}) mm")
    if actual_entry:
        fx, fy = actual_entry["final_xy_mm"]
        px, py = actual_entry["planned_xy_mm"]
        dx, dy = fx - px, fy - py
        print_kv("    Final XY",    f"({fx:.2f}, {fy:.2f}) mm")
        print_kv("    PD offset",   f"dX={dx:+.2f}  dY={dy:+.2f} mm")
    print()


def print_skip_target(i: int, total: int, solved_target, reason: str):
    _finalize_fine_line()
    tx, ty = solved_target["target_xy_mm"]
    print(_yellow(f"  ⚠ Target {i}/{total} skipped") + _gray(f"  ({tx:.1f}, {ty:.1f}) mm"))
    print(_gray(f"    {reason}"))
    print()


# ── fine-align live status ────────────────────────────────────────────────────

def print_live_fine_align(
    err_x: float, err_y: float,
    dx: float, dy: float,
    planned_xy=None,
    throttle_s: float = 0.15,
    settle_count: int = 0,
    settle_frames: int = 0,
    elapsed_s: float = None,
    max_time_s: float = None,
):
    """
    Overwrites a single terminal line in-place while fine-align is running.
    Falls back to a normal print when not on a TTY (log files, piped output).
    """
    global _fine_in_place, _last_fine_time

    now = time.time()
    if now - _last_fine_time < throttle_s:
        return
    _last_fine_time = now

    mag = abs(err_x) + abs(err_y)
    if mag < 3:
        err_col = _green
    elif mag < 15:
        err_col = _yellow
    else:
        err_col = _red

    err_str = err_col(f"({err_x:+6.1f}, {err_y:+6.1f})px")
    jog_str = _cyan(f"({dx:+.4f}, {dy:+.4f})mm")

    settle_str = ""
    if settle_frames > 0:
        settle_str = f"  settle {_green(str(settle_count))}/{settle_frames}"

    elapsed_str = ""
    if elapsed_s is not None:
        if max_time_s is not None:
            elapsed_str = f" {elapsed_s:4.1f}/{max_time_s:.0f}s"
        else:
            elapsed_str = f" {elapsed_s:4.1f}s"

    line = f"  ⟳ Fine{elapsed_str}  err {err_str}  jog {jog_str}{settle_str}"

    if _IS_TTY:
        w       = _w()
        # Strip ANSI codes for length calc is complex; just truncate raw
        padded  = line.ljust(w - 1)[:w - 1]
        sys.stdout.write(f"\r{padded}")
        sys.stdout.flush()
        _fine_in_place = True
    else:
        print(line)


def end_live_fine_align(status: str = ""):
    """Call when fine-align finishes (success or failure) to close the live line."""
    global _fine_in_place
    if _fine_in_place and _IS_TTY:
        suffix = f"  {status}" if status else ""
        sys.stdout.write(f"\r  ✓ Fine align done{suffix}\n")
        sys.stdout.flush()
        _fine_in_place = False
    else:
        msg = f"Fine align: {status}" if status else "Fine align done"
        print(f"  {msg}")
