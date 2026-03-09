import sys


def print_match_summary(matched_targets, unmatched_left, unmatched_right):
    print("\n" + "=" * 70)
    print("MATCH SUMMARY")
    print("=" * 70)

    print(f"Matched targets : {len(matched_targets)}")
    print(f"Unmatched left  : {len(unmatched_left)}")
    print(f"Unmatched right : {len(unmatched_right)}")
    print("-" * 70)

    if matched_targets:
        print("IDX | LEFT (x,y)      | RIGHT (x,y)     | SCORE")
        print("-" * 70)

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

    print("=" * 70 + "\n")


def print_workspace_plan(absolute_targets):
    print("\n" + "=" * 70)
    print("PLANNED WORKSPACE TARGETS")
    print("=" * 70)

    if not absolute_targets:
        print("No targets planned.")
        print("=" * 70 + "\n")
        return

    print("IDX | X (mm)     | Y (mm)")
    print("-" * 70)
    for i, solved in enumerate(absolute_targets, start=1):
        tx, ty = solved["target_xy_mm"]
        print(f"{i:>3} | {tx:>10.2f} | {ty:>10.2f}")

    print("=" * 70 + "\n")


def show_current_target(i, total, solved_target):
    tx, ty = solved_target["target_xy_mm"]
    message = f"Current target {i}/{total}: X={tx:.2f} mm, Y={ty:.2f} mm"
    sys.stdout.write("\r\033[2K" + message)
    sys.stdout.flush()


def clear_current_target_line():
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()
