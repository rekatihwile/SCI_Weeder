#!/usr/bin/env python3
"""Arrow-key teleop for the Scout.

Controls:
  ↑ / ↓    forward / reverse
  ← / →    turn left / right
  space    hard stop
  q        quit

Deadman: if no key arrives within DEADMAN_S, target zeroes out.
Hold an arrow → terminal auto-repeat keeps it moving.
Tap → moves briefly then stops.
"""
import select
import sys
import termios
import time
import tty

from scout import Scout

# tuneables
LIN_STEP  = 0.1   # m/s per up/down press
ANG_STEP  = 0.1   # rad/s per left/right press
DEADMAN_S = 0.25   # auto-stop if no key in this window
LOOP_HZ   = 20

ARROW = {"A": "up", "B": "down", "C": "right", "D": "left"}


def read_key(timeout):
    """Non-blocking single key read; returns 'up'/'down'/'left'/'right'/'space'/'q' or None."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":                 # ESC — start of arrow seq
        seq = sys.stdin.read(2)
        if len(seq) == 2 and seq[0] == "[" and seq[1] in ARROW:
            return ARROW[seq[1]]
        return None
    if ch == " ":
        return "space"
    if ch in ("q", "Q", "\x03"):     # q or Ctrl-C
        return "q"
    return None


def main():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    print(__doc__)
    print("connecting to scout...")
    with Scout(model="mini") as scout:   # change to "scout2" if you want
        s = scout.state()
        print(f"battery {s['battery_v']:.2f} V   err {s['error']}   mode {s['ctrl_mode']}\n")

        try:
            tty.setcbreak(fd)
            lin = ang = 0.0
            last_key_t = time.monotonic()
            period = 1.0 / LOOP_HZ

            while True:
                t = time.monotonic()
                key = read_key(timeout=period)

                if key == "q":
                    break
                elif key == "space":
                    lin = ang = 0.0
                    last_key_t = t
                elif key == "up":
                    lin, ang = +LIN_STEP, 0.0; last_key_t = t
                elif key == "down":
                    lin, ang = -LIN_STEP, 0.0; last_key_t = t
                elif key == "left":
                    lin, ang = 0.0, +ANG_STEP; last_key_t = t
                elif key == "right":
                    lin, ang = 0.0, -ANG_STEP; last_key_t = t

                # deadman
                if t - last_key_t > DEADMAN_S:
                    lin = ang = 0.0

                scout.drive(lin, ang)

                st = scout.state()
                sys.stdout.write(
                    f"\rcmd lin={lin:+.2f} ang={ang:+.2f}  |  "
                    f"actual lin={st['linear']:+.2f} ang={st['angular']:+.2f}  "
                    f"batt {st['battery_v']:.1f}V    "
                )
                sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print("\nstopped.")


if __name__ == "__main__":
    main()