"""
bringup/03_gantry_status.py
----------------------------
Read GRBL status only. No motion. No homing. No reset.
Opens serial directly using GRBL_PORT.

Run with:
    ./run_with_eli_venv.sh bringup/03_gantry_status.py | tee bringup/logs/03_gantry_status.log
"""

import sys
import time
import re
import serial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_lines(ser, duration=0.5):
    """Read all available lines from serial for `duration` seconds."""
    lines = []
    end_time = time.time() + duration
    while time.time() < end_time:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            lines.append(line)
    return lines


def send_query(ser, cmd_bytes, label, duration=0.5):
    """Send a raw command and print all response lines."""
    print(f"\n--- {label} ---")
    ser.reset_input_buffer()
    ser.write(cmd_bytes)
    lines = read_lines(ser, duration)
    for line in lines:
        print(f"  {line}")
    return lines


def send_command(ser, cmd_str, label, duration=0.5):
    """Send a text command terminated by newline and print all response lines."""
    print(f"\n--- {label} ---")
    ser.reset_input_buffer()
    ser.write((cmd_str.strip() + "\n").encode())
    lines = read_lines(ser, duration)
    for line in lines:
        print(f"  {line}")
    return lines


def parse_position(status_line):
    """Parse MPos and WPos from a GRBL status line like <Idle|MPos:0.000,0.000,0.000|...>"""
    mpos = wpos = None
    m = re.search(r"MPos:([-\d.]+),([-\d.]+),([-\d.]+)", status_line)
    if m:
        mpos = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = re.search(r"WPos:([-\d.]+),([-\d.]+),([-\d.]+)", status_line)
    if m:
        wpos = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return mpos, wpos


def main():
    from config import GRBL_PORT

    print("=" * 60)
    print("BRINGUP 03 — Gantry Status")
    print("=" * 60)
    print(f"\n  GRBL_PORT: {GRBL_PORT}")

    print(f"\n--- Opening serial: {GRBL_PORT} @ 115200 ---")
    ser = serial.Serial(GRBL_PORT, 115200, timeout=0.5)
    time.sleep(2.0)  # GRBL reset grace period
    ser.reset_input_buffer()
    print("  Serial opened.")

    # Read any startup banner
    banner = read_lines(ser, 0.5)
    if banner:
        print("  Startup banner:")
        for line in banner:
            print(f"    {line}")

    # ? — status query
    status_lines = send_query(ser, b"?", "? (status)")
    mpos = wpos = None
    for line in status_lines:
        if line.startswith("<"):
            m, w = parse_position(line)
            if m:
                mpos = m
            if w:
                wpos = w

    print("\n--- Parsed position ---")
    if mpos:
        print(f"  MPos: x={mpos[0]:.3f}  y={mpos[1]:.3f}  z={mpos[2]:.3f}")
    else:
        print("  MPos: not found in response")
    if wpos:
        print(f"  WPos: x={wpos[0]:.3f}  y={wpos[1]:.3f}  z={wpos[2]:.3f}")

    # $G — modal state
    send_command(ser, "$G", "$G (modal state)", duration=0.5)

    # $# — coordinate system offsets
    send_command(ser, "$#", "$# (coordinate offsets)", duration=0.5)

    # $$ — GRBL settings
    send_command(ser, "$$", "$$ (settings)", duration=1.0)

    ser.close()
    print("\n  Serial closed.")

    received = len(status_lines) > 0
    print("\n" + "=" * 60)
    if received:
        print("RESULT: PASS  (serial opened and status response received)")
    else:
        print("RESULT: FAIL  (no status response from GRBL)")
    print("=" * 60)

    sys.exit(0 if received else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
