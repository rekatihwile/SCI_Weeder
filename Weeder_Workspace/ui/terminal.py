import os
import sys
import time

_last_fine_line = ""
_last_fine_time = 0.0


def _term_width(default=90):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _rule(char="=", width=None):
    if width is None:
        width = _term_width()
    return char * width


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title):
    width = _term_width()
    print(_rule("=", width))
    print(title.center(width))
    print(_rule("=", width))


def print_section(title):
    width = _term_width()
    print("\n" + _rule("-", width))
    print(title)
    print(_rule("-", width))


def print_kv(label, value, indent=0):
    pad = " " * indent
    print(f"{pad}{label:<32} {value}")


def print_status_banner(state, subtitle=None):
    clear_screen()
    print_header("LASER WEEDER")
    print_kv("STATE", state)
    if subtitle:
        print_kv("INFO", subtitle)


def print_match_summary(matched_targets, unmatched_left, unmatched_right):
    width = _term_width()
    print("\n" + _rule("=", width))
    print("MATCH SUMMARY")
    print(_rule("=", width))

    print_kv("Matched targets", len(matched_targets))
    print_kv("Unmatched left", len(unmatched_left))
    print_kv("Unmatched right", len(unmatched_right))
    print(_rule("-", width))

    if matched_targets:
        print("IDX | LEFT (x,y)      | RIGHT (x,y)     | SCORE")
        print(_rule("-", width))

        for i, t in enumerate(matched_targets):
            lp = t["left_px"]
            rp = t["right_px"]
            sc = t["score"]
            print(f"{i:>3} | {str(lp):<15} | {str(rp):<15} | {sc:0.3f}")

    if unmatched_left:
        print("\nUnmatched LEFT:")
        print(", ".join(str(p) for p in unmatched_left))

    if unmatched_right:
        print("\nUnmatched RIGHT:")
        print(", ".join(str(p) for p in unmatched_right))

    print(_rule("=", width) + "\n")


def print_workspace_plan(absolute_targets):
    width = _term_width()
    print("\n" + _rule("=", width))
    print("PLANNED WORKSPACE TARGETS")
    print(_rule("=", width))

    if not absolute_targets:
        print("No targets planned.")
        print(_rule("=", width) + "\n")
        return

    print("IDX | X (mm)     | Y (mm)")
    print(_rule("-", width))
    for i, solved in enumerate(absolute_targets, start=1):
        tx, ty = solved["target_xy_mm"]
        print(f"{i:>3} | {tx:>10.2f} | {ty:>10.2f}")

    print(_rule("=", width) + "\n")


def show_current_target(i, total, solved_target):
    tx, ty = solved_target["target_xy_mm"]

    clear_screen()
    print_header("LASER WEEDER - ACTIVE TARGET")

    print_kv("Target", f"{i}/{total}")
    print_kv("Planned X (mm)", f"{tx:.2f}")
    print_kv("Planned Y (mm)", f"{ty:.2f}")

    if "source_target" in solved_target:
        src = solved_target["source_target"]
        if "left_px" in src and "right_px" in src:
            print_kv("Left image point", src["left_px"])
            print_kv("Right image point", src["right_px"])
        if "score" in src:
            print_kv("Match score", f"{src['score']:.3f}")

    print("\nWaiting for coarse move / fine align...")


def print_target_result(i, total, solved_target, actual_entry=None):
    tx, ty = solved_target["target_xy_mm"]

    clear_screen()
    print_header("LASER WEEDER - TARGET RESULT")

    print_kv("Target", f"{i}/{total}")
    print_kv("Coarse Triangulation Coordinates", f"X={tx:.2f} mm, Y={ty:.2f} mm")

    if actual_entry is not None:
        fx, fy = actual_entry["final_xy_mm"]
        print_kv("PD Confirmed Coordinates", f"X={fx:.2f} mm, Y={fy:.2f} mm")

        if "selected_local_xy_mm" in actual_entry:
            sx, sy = actual_entry["selected_local_xy_mm"]
            print_kv("Local Re-triangulated Choice", f"X={sx:.2f} mm, Y={sy:.2f} mm")

        px, py = actual_entry["planned_xy_mm"]
        dx = fx - px
        dy = fy - py
        print_kv("PD Offset from Planned", f"dX={dx:.2f} mm, dY={dy:.2f} mm")

    else:
        print_kv("PD Confirmed Coordinates", "Not available")

    print("\nTarget completed.\n")


def print_skip_target(i, total, solved_target, reason):
    tx, ty = solved_target["target_xy_mm"]

    clear_screen()
    print_header("LASER WEEDER - TARGET SKIPPED")
    print_kv("Target", f"{i}/{total}")
    print_kv("Planned Coordinates", f"X={tx:.2f} mm, Y={ty:.2f} mm")
    print_kv("Reason", reason)
    print()


def print_global_survey_ready(x, y):
    clear_screen()
    print_header("LASER WEEDER - GLOBAL SURVEY")
    print_kv("Survey Position", f"X={x:.2f} mm, Y={y:.2f} mm")
    print()
    print("Enter = survey | q = quit")
    print()


def print_global_survey_results(left_count, right_count, matched_count):
    print_section("GLOBAL SURVEY RESULTS")
    print_kv("Stable left points", left_count)
    print_kv("Stable right points", right_count)
    print_kv("Matched targets", matched_count)
    print()


def print_live_fine_align(err_x, err_y, dx, dy, planned_xy=None, throttle_s=0.20):
    global _last_fine_line, _last_fine_time

    now = time.time()
    if now - _last_fine_time < throttle_s:
        return

    if planned_xy is not None:
        px, py = planned_xy
        line = (
            f"\rFine Align | coarse=({px:.2f}, {py:.2f}) mm | "
            f"err=({err_x:.2f}, {err_y:.2f}) px | "
            f"jog=({dx:.3f}, {dy:.3f}) mm"
        )
    else:
        line = (
            f"\rFine Align | err=({err_x:.2f}, {err_y:.2f}) px | "
            f"jog=({dx:.3f}, {dy:.3f}) mm"
        )

    width = _term_width()
    padded = line.ljust(width - 1)

    sys.stdout.write(padded)
    sys.stdout.flush()

    _last_fine_line = padded
    _last_fine_time = now


def end_live_fine_align():
    sys.stdout.write("\n")
    sys.stdout.flush()


def clear_current_target_line():
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()