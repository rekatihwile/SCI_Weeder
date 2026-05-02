"""Gantry lifecycle + command helpers for the remote dashboard.

This module wraps the real hardware.gantry.Gantry class.

It owns:
- opening/closing the shared gantry serial connection
- reading gantry status/position
- sending dashboard commands
- sending raw GRBL/G-code commands

Routes live in dashboard_routes.py.
"""

from config import GRBL_PORT
from hardware.gantry import Gantry
from dashboard_state import state


# =============================================================================
# Lifecycle
# =============================================================================

def ensure_gantry():
    """Open the real gantry connection if it is not already open."""
    if state.gantry is None:
        state.gantry = Gantry(GRBL_PORT)
    return state.gantry


def close_gantry():
    """Close the real gantry connection and clear shared state."""
    if state.gantry is not None:
        try:
            state.gantry.close()
        finally:
            state.gantry = None


# =============================================================================
# Serial/status helpers
# =============================================================================

def _read_extra_lines(g, duration=0.25):
    """Collect extra serial lines if the Gantry class exposes that helper."""
    if hasattr(g, "read_all_available_lines"):
        try:
            lines = g.read_all_available_lines(duration=duration)
            return lines if lines is not None else []
        except Exception:
            return []
    return []


def _send_raw(g, cmd, duration=0.25):
    """Send a raw command and collect immediate + extra responses."""
    first = g.send_raw(cmd)
    extra = _read_extra_lines(g, duration=duration)
    return [x for x in [first, *extra] if x]


def _safe_status(g):
    """Return JSON-safe gantry status without raising if one field fails."""
    payload = {
        "connected": True,
        "port": GRBL_PORT,
    }

    try:
        payload["status_line"] = g.get_status_line()
    except Exception as e:
        payload["status_line_error"] = repr(e)

    try:
        payload["position"] = g.get_position()
    except Exception as e:
        payload["position_error"] = repr(e)

    try:
        payload["estimated_position"] = g.get_estimated_position()
    except Exception as e:
        payload["estimated_position_error"] = repr(e)

    try:
        payload["modal"] = _send_raw(g, "$G", duration=0.15)
    except Exception as e:
        payload["modal_error"] = repr(e)

    return payload


# =============================================================================
# Public payloads
# =============================================================================

def gantry_status_payload():
    """Open gantry if needed, then return a broad status payload."""
    g = ensure_gantry()
    return _safe_status(g)


def gantry_position_payload():
    """Return where GRBL and the Gantry object think the machine is."""
    g = ensure_gantry()

    payload = {
        "connected": True,
        "port": GRBL_PORT,
        "message": "Position read complete.",
    }

    try:
        payload["status_line"] = g.get_status_line()
    except Exception as e:
        payload["status_line_error"] = repr(e)

    try:
        payload["position"] = g.get_position()
    except Exception as e:
        payload["position_error"] = repr(e)

    try:
        payload["estimated_position"] = g.get_estimated_position()
    except Exception as e:
        payload["estimated_position_error"] = repr(e)

    try:
        payload["modal"] = _send_raw(g, "$G", duration=0.15)
    except Exception as e:
        payload["modal_error"] = repr(e)

    try:
        payload["offsets"] = _send_raw(g, "$#", duration=0.25)
    except Exception as e:
        payload["offsets_error"] = repr(e)

    return payload


# =============================================================================
# Dashboard commands
# =============================================================================

def unlock_gantry():
    """Send GRBL unlock command."""
    g = ensure_gantry()
    responses = _send_raw(g, "$X", duration=0.35)

    payload = _safe_status(g)
    payload["command"] = "$X"
    payload["responses"] = responses
    payload["message"] = "Unlock command sent."
    return payload


def reset_gantry():
    """Send soft reset through Gantry class if available."""
    g = ensure_gantry()

    if not hasattr(g, "soft_reset"):
        raise RuntimeError("Gantry.soft_reset() not found.")

    g.soft_reset()

    payload = _safe_status(g)
    payload["command"] = "soft_reset"
    payload["message"] = "Soft reset sent."
    return payload


def stop_gantry():
    """Send stop through Gantry class if available."""
    g = ensure_gantry()

    if not hasattr(g, "stop"):
        raise RuntimeError("Gantry.stop() not found.")

    g.stop()

    payload = _safe_status(g)
    payload["command"] = "stop"
    payload["message"] = "Stop command sent."
    return payload


def home_gantry():
    """Run the real homing sequence."""
    g = ensure_gantry()
    g.home()

    payload = _safe_status(g)
    payload["command"] = "home"
    payload["message"] = "Homing complete."
    return payload


def jog_gantry(dx, dy, feed=None):
    """Jog the gantry by relative dx, dy in mm."""
    g = ensure_gantry()

    if not hasattr(g, "jog"):
        raise RuntimeError("Gantry.jog() not found.")

    if feed is None:
        try:
            g.jog(dx, dy)
        except TypeError:
            g.jog(dx=dx, dy=dy)
    else:
        try:
            g.jog(dx, dy, feed=feed)
        except TypeError:
            try:
                g.jog(dx=dx, dy=dy, feed=feed)
            except TypeError:
                g.jog(dx, dy)

    payload = _safe_status(g)
    payload["command"] = "jog"
    payload["dx_mm"] = dx
    payload["dy_mm"] = dy
    payload["feed"] = feed
    payload["message"] = "Jog command sent."
    return payload


def move_absolute_gantry(x, y, feed=None):
    """Move gantry to absolute X,Y in mm."""
    g = ensure_gantry()

    if not hasattr(g, "move_absolute"):
        raise RuntimeError("Gantry.move_absolute() not found.")

    if feed is None:
        try:
            g.move_absolute(x, y)
        except TypeError:
            g.move_absolute(x=x, y=y)
    else:
        try:
            g.move_absolute(x, y, feed=feed)
        except TypeError:
            try:
                g.move_absolute(x=x, y=y, feed=feed)
            except TypeError:
                g.move_absolute(x, y)

    payload = _safe_status(g)
    payload["command"] = "move_absolute"
    payload["target_x_mm"] = x
    payload["target_y_mm"] = y
    payload["feed"] = feed
    payload["message"] = "Absolute move command sent."
    return payload


def raw_gcode(cmd):
    """Send raw GRBL/G-code command string."""
    cmd = (cmd or "").strip()
    if not cmd:
        raise ValueError("Raw command is empty.")

    g = ensure_gantry()
    responses = _send_raw(g, cmd, duration=0.5)

    payload = _safe_status(g)
    payload["command"] = cmd
    payload["responses"] = responses
    payload["message"] = "Raw command sent."
    return payload