#!/usr/bin/env python3
"""Snappy WASD/arrow teleop for the Scout, with pose tracking and waypoints.

DRIVE KEYS:
  w / s   or  ↑ / ↓     forward / reverse
  a / d   or  ← / →     turn left / right
  space                 hard stop (also cancels :goto)
  + / -                 bump cruise speed
  q                     quit

COMMAND MODE (press `:` or `/` to enter, vim-style):
  :save NAME    save current pose
  :goto NAME    drive back to NAME (space/drive key cancels)
  :home         drive back to (0, 0, 0)
  :list         list waypoints with distances
  :del NAME     delete waypoint
  :reset        reset pose to origin
  :help         re-print this header
  esc           leave command mode

CLI:
  --accel [A]          enable smooth ramping (default 0.8 m/s²)
  --ang-accel A        angular accel rad/s² (default 1.5, implies --accel)
  --speed S            initial linear cruise (m/s, default 0.10)
  --turn-rate T        initial angular cruise (rad/s, default 0.10)
  --model {mini,scout2}

Per-session CSV log: ./scout_logs/scout_YYYYMMDD_HHMMSS.csv
Waypoints persist: ./scout_waypoints.json
"""
import argparse, json, math, select, shutil, sys, termios, time, tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scout import Scout

DEADMAN_S     = 0.25
LOOP_HZ       = 20
ESC_TIMEOUT_S = 0.10   # generous so arrow keys arrive intact over SSH

GOTO_DIST_TOL   = 0.10
GOTO_ANG_TOL    = 0.08
GOTO_V_MAX      = 0.30
GOTO_W_MAX      = 0.50
GOTO_K_DIST     = 0.8
GOTO_K_HEAD     = 1.8
GOTO_TURN_FIRST = 0.35
GOTO_TIMEOUT_S  = 120.0

WAYPOINTS_PATH = Path("scout_waypoints.json")
LOG_DIR        = Path("scout_logs")
ARROW = {"A": "up", "B": "down", "C": "right", "D": "left"}

CANCEL_DRIVE_KEYS = {"space", "up", "down", "left", "right",
                     "w", "a", "s", "d", "W", "A", "S", "D"}


def wrap_pi(a): return (a + math.pi) % (2 * math.pi) - math.pi
def clamp(x, lo, hi): return max(lo, min(hi, x))


@dataclass
class Pose:
    x: float = 0.0; y: float = 0.0; theta: float = 0.0
    def update(self, v, w, dt):
        th_mid = self.theta + 0.5 * w * dt
        self.x += v * math.cos(th_mid) * dt
        self.y += v * math.sin(th_mid) * dt
        self.theta = wrap_pi(self.theta + w * dt)


def load_waypoints():
    if WAYPOINTS_PATH.exists():
        try:
            return {k: tuple(v) for k, v in json.loads(WAYPOINTS_PATH.read_text()).items()}
        except Exception:
            return {}
    return {}

def save_waypoints(w):
    WAYPOINTS_PATH.write_text(json.dumps({k: list(v) for k, v in w.items()}, indent=2))


def read_input(timeout):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r: return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        r2, _, _ = select.select([sys.stdin], [], [], ESC_TIMEOUT_S)
        if not r2: return "esc"
        ch2 = sys.stdin.read(1)
        if ch2 != "[": return None
        r3, _, _ = select.select([sys.stdin], [], [], ESC_TIMEOUT_S)
        if not r3: return None
        return ARROW.get(sys.stdin.read(1))
    if ch in ("\r", "\n"):   return "enter"
    if ch in ("\x7f", "\b"): return "backspace"
    if ch == " ":            return "space"
    if ch == "\x03":         return "ctrl_c"
    return ch


def status_print(text):
    cols = shutil.get_terminal_size((80, 24)).columns - 1
    sys.stdout.write("\r\x1b[K" + text[:cols])
    sys.stdout.flush()

def announce(msg):
    sys.stdout.write("\r\x1b[K" + msg + "\n")
    sys.stdout.flush()


def execute_goto(scout, pose, target, log_writer):
    tx, ty, tt = target
    period = 1.0 / LOOP_HZ
    t0 = time.monotonic()
    last_t = t0
    announce(f"→ goto ({tx:+.2f}, {ty:+.2f}, {math.degrees(tt or 0):+.0f}°)  "
             f"space/drive key cancels")

    while True:
        t = time.monotonic()
        dt = max(1e-3, t - last_t); last_t = t

        st = scout.state()
        pose.update(st["linear"], st["angular"], dt)
        log_writer(t, pose, st["linear"], st["angular"])

        key = read_input(0.0)
        if key in CANCEL_DRIVE_KEYS or key in ("q", "esc", "ctrl_c"):
            scout.stop()
            announce(f"  cancelled at ({pose.x:+.2f}, {pose.y:+.2f}, "
                     f"{math.degrees(pose.theta):+.0f}°)")
            return key

        dx, dy = tx - pose.x, ty - pose.y
        dist = math.hypot(dx, dy)

        if dist < GOTO_DIST_TOL:
            if tt is None: scout.stop(); break
            head_err = wrap_pi(tt - pose.theta)
            if abs(head_err) < GOTO_ANG_TOL: scout.stop(); break
            scout.drive(0.0, clamp(GOTO_K_HEAD * head_err, -GOTO_W_MAX, GOTO_W_MAX))
        else:
            target_heading = math.atan2(dy, dx)
            head_err = wrap_pi(target_heading - pose.theta)
            if dist > 0.4 and abs(head_err) > GOTO_TURN_FIRST:
                scout.drive(0.0, clamp(GOTO_K_HEAD * head_err, -GOTO_W_MAX, GOTO_W_MAX))
            else:
                v = clamp(GOTO_K_DIST * dist, 0.05, GOTO_V_MAX)
                w = clamp(GOTO_K_HEAD * head_err, -GOTO_W_MAX, GOTO_W_MAX)
                scout.drive(v, w)

        status_print(
            f"goto d={dist:.2f}m herr={math.degrees(head_err):+.0f}°  "
            f"pose({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.theta):+.0f}°)  "
            f"t={t-t0:.1f}s"
        )

        slept = time.monotonic() - t
        if slept < period: time.sleep(period - slept)
        if t - t0 > GOTO_TIMEOUT_S:
            scout.stop(); announce("  goto timed out."); return None

    announce(f"  reached ({pose.x:+.2f}, {pose.y:+.2f}, "
             f"{math.degrees(pose.theta):+.0f}°)")
    return None


def handle_command(line, pose, waypoints, scout, log_writer):
    line = line.strip()
    if not line: return None
    parts = line.split(maxsplit=1)
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd in ("h", "help", "?"):
        announce(__doc__)
    elif cmd == "save":
        if not arg: announce("  usage: :save NAME"); return None
        waypoints[arg] = (pose.x, pose.y, pose.theta)
        save_waypoints(waypoints)
        announce(f"  saved {arg!r} → ({pose.x:+.2f}, {pose.y:+.2f}, "
                 f"{math.degrees(pose.theta):+.0f}°)")
    elif cmd in ("goto", "g"):
        if not arg: announce("  usage: :goto NAME"); return None
        if arg not in waypoints: announce(f"  no such waypoint: {arg!r}"); return None
        r = execute_goto(scout, pose, waypoints[arg], log_writer)
        if r in ("q", "ctrl_c"): return "quit"
        if r in CANCEL_DRIVE_KEYS: return r
    elif cmd == "home":
        r = execute_goto(scout, pose, (0.0, 0.0, 0.0), log_writer)
        if r in ("q", "ctrl_c"): return "quit"
        if r in CANCEL_DRIVE_KEYS: return r
    elif cmd in ("list", "ls", "l"):
        if not waypoints: announce("  (no waypoints saved)")
        else:
            for name, (x, y, th) in sorted(waypoints.items()):
                d = math.hypot(x - pose.x, y - pose.y)
                announce(f"  {name:20s} ({x:+.2f}, {y:+.2f}, "
                         f"{math.degrees(th):+.0f}°)   {d:.2f}m away")
    elif cmd in ("del", "rm", "delete"):
        if not arg: announce("  usage: :del NAME"); return None
        if arg not in waypoints: announce(f"  no such waypoint: {arg!r}"); return None
        del waypoints[arg]; save_waypoints(waypoints)
        announce(f"  deleted {arg!r}")
    elif cmd == "reset":
        pose.x = pose.y = pose.theta = 0.0
        announce("  pose reset to origin")
    elif cmd in ("q", "quit", "exit"):
        return "quit"
    else:
        announce(f"  unknown command: {cmd!r}  (try :help)")
    return None


def main():
    ap = argparse.ArgumentParser(description="Scout teleop")
    ap.add_argument("--accel", type=float, nargs="?", const=0.8, default=None,
                    metavar="A", help="enable smooth ramping (default 0.8 m/s²)")
    ap.add_argument("--ang-accel", type=float, default=None, metavar="A",
                    help="angular accel rad/s² (implies --accel)")
    ap.add_argument("--speed", type=float, default=0.10, metavar="S",
                    help="initial linear cruise m/s (default 0.10)")
    ap.add_argument("--turn-rate", type=float, default=0.10, metavar="T",
                    help="initial angular cruise rad/s (default 0.10)")
    ap.add_argument("--model", choices=["mini", "scout2"], default="mini")
    args = ap.parse_args()

    smooth = args.accel is not None or args.ang_accel is not None
    lin_accel = args.accel if args.accel is not None else 0.8
    ang_accel = args.ang_accel if args.ang_accel is not None else 1.5

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_path, "w", buffering=1)
    log_file.write("t,x,y,theta,v,w\n")
    t_start = time.monotonic()

    def log_writer(t, p, v, w):
        log_file.write(f"{t-t_start:.3f},{p.x:.4f},{p.y:.4f},"
                       f"{p.theta:.4f},{v:.3f},{w:.3f}\n")

    waypoints = load_waypoints()
    print(__doc__)
    mode_str = (f"smooth (accel {lin_accel}/{ang_accel})"
                if smooth else "snappy (bang-bang, no ramping)")
    print(f"mode:      {mode_str}")
    print(f"log:       {log_path}")
    print(f"waypoints: {len(waypoints)} loaded")
    print("connecting to scout...")

    pose = Pose()

    with Scout(model=args.model) as scout:
        s = scout.state()
        print(f"battery {s['battery_v']:.2f} V   err {s['error']}   "
              f"mode {s['ctrl_mode']}")
        print("ready — w/a/s/d (or arrows) to drive, : for command, q quit\n")

        try:
            tty.setcbreak(fd)

            lin = ang = 0.0
            target_lin = target_ang = 0.0
            lin_step = args.speed
            ang_step = args.turn_rate
            last_key_t = time.monotonic()
            last_t = last_key_t
            period = 1.0 / LOOP_HZ

            mode = "drive"
            cmd_buf = ""
            pending_key = None
            last_key_name = "-"
            last_key_real_t = time.monotonic() - 99.0

            while True:
                t = time.monotonic()
                dt = max(1e-3, t - last_t); last_t = t

                if pending_key is not None:
                    key, pending_key = pending_key, None
                else:
                    key = read_input(timeout=period)

                if key is not None:
                    last_key_name = {"up":"↑","down":"↓","left":"←","right":"→",
                                     "space":"spc","enter":"↵","esc":"esc",
                                     "backspace":"bs"}.get(key, str(key))
                    last_key_real_t = t

                if mode == "drive":
                    if key in ("q", "ctrl_c"): break
                    elif key in (":", "/"):
                        mode = "command"; cmd_buf = ""
                        target_lin = target_ang = 0.0
                    elif key == "space":
                        target_lin = target_ang = 0.0; last_key_t = t
                    elif key in ("up", "w", "W"):
                        target_lin, target_ang = +lin_step, 0.0; last_key_t = t
                    elif key in ("down", "s", "S"):
                        target_lin, target_ang = -lin_step, 0.0; last_key_t = t
                    elif key in ("left", "a", "A"):
                        target_lin, target_ang = 0.0, +ang_step; last_key_t = t
                    elif key in ("right", "d", "D"):
                        target_lin, target_ang = 0.0, -ang_step; last_key_t = t
                    elif key in ("+", "="):
                        lin_step = min(scout.MAX_LIN, lin_step + 0.05)
                        ang_step = min(scout.MAX_ANG, ang_step + 0.05)
                    elif key in ("-", "_"):
                        lin_step = max(0.05, lin_step - 0.05)
                        ang_step = max(0.05, ang_step - 0.05)
                    elif key in ("?", "h"):
                        announce(__doc__)

                    if t - last_key_t > DEADMAN_S:
                        target_lin = target_ang = 0.0

                    if smooth:
                        lin += clamp(target_lin - lin, -lin_accel * dt, lin_accel * dt)
                        ang += clamp(target_ang - ang, -ang_accel * dt, ang_accel * dt)
                    else:
                        lin, ang = target_lin, target_ang

                    scout.drive(lin, ang)

                    st = scout.state()
                    pose.update(st["linear"], st["angular"], dt)
                    log_writer(t, pose, st["linear"], st["angular"])

                    age = t - last_key_real_t
                    age_str = f"{age:.1f}s" if age < 99 else "—"
                    status_print(
                        f"cmd v={lin:+.2f} w={ang:+.2f}  "
                        f"act v={st['linear']:+.2f} w={st['angular']:+.2f}  "
                        f"pose({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.theta):+.0f}°)  "
                        f"sp={lin_step:.2f}  bat {st['battery_v']:.1f}V  "
                        f"k:{last_key_name}({age_str})"
                    )

                else:  # command mode
                    target_lin = target_ang = 0.0
                    if smooth:
                        lin += clamp(-lin, -lin_accel * dt, lin_accel * dt)
                        ang += clamp(-ang, -ang_accel * dt, ang_accel * dt)
                    else:
                        lin = ang = 0.0
                    scout.drive(lin, ang)

                    if key == "enter":
                        result = handle_command(cmd_buf, pose, waypoints,
                                                scout, log_writer)
                        cmd_buf = ""; mode = "drive"
                        last_key_t = time.monotonic()
                        if result == "quit": break
                        if result in CANCEL_DRIVE_KEYS:
                            pending_key = result
                    elif key in ("esc", "ctrl_c"):
                        cmd_buf = ""; mode = "drive"
                        announce("  (cancelled)")
                    elif key == "backspace":
                        cmd_buf = cmd_buf[:-1]
                    elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                        cmd_buf += key

                    status_print(f": {cmd_buf}_")

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            try: scout.stop()
            except Exception: pass
            log_file.close()
            sys.stdout.write("\n")
            print(f"stopped. log: {log_path}")
            print(f"final pose: ({pose.x:+.2f}, {pose.y:+.2f}, "
                  f"{math.degrees(pose.theta):+.0f}°)")


if __name__ == "__main__":
    main()