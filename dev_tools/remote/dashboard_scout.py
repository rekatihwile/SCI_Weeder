"""Scout lifecycle + command helpers for the remote dashboard.

Wraps ScoutController for use in dashboard routes.

Owns:
- opening/closing the shared Scout CAN connection
- check_connection, stop, move_forward/move_backward actions

Routes live in dashboard_routes.py.
"""

from config import (
    SCOUT_ENABLED,
    SCOUT_CAN_INTERFACE,
    SCOUT_DRY_RUN,
    SCOUT_ADVANCE_DISTANCE_M,
    SCOUT_ADVANCE_SPEED_MPS,
    SCOUT_ADVANCE_TIMEOUT_SEC,
)
from dashboard_state import state


# =============================================================================
# Lifecycle
# =============================================================================

def ensure_scout():
    """Open Scout CAN connection if not already open."""
    if state.scout is None:
        from scout.scout_controller import ScoutController
        state.scout = ScoutController(
            interface=SCOUT_CAN_INTERFACE,
            dry_run=SCOUT_DRY_RUN,
        )
    return state.scout


def close_scout():
    """Stop Scout and close the shared connection."""
    if state.scout is not None:
        try:
            state.scout.close()
        finally:
            state.scout = None


# =============================================================================
# Dashboard actions
# =============================================================================

def scout_check_connection() -> dict:
    """Check CAN link and Scout responsiveness. Updates last_scout_check."""
    sc = ensure_scout()
    result = sc.check_connection()
    state.last_scout_check = result
    return result


def scout_stop() -> dict:
    """Send stop command. Updates last_scout_move."""
    sc = ensure_scout()
    try:
        sc.stop()
        result = {"ok": True, "message": "Stop command sent."}
    except Exception as exc:
        result = {"ok": False, "message": f"Stop failed: {exc}"}
    state.last_scout_move = result
    return result


def scout_move_forward(
    distance_m: float,
    speed_mps: float,
    timeout_s: float = None,
    dry_run: bool = False,
) -> dict:
    """Command a forward move. Updates last_scout_move."""
    if dry_run:
        # Override the controller's dry_run flag for this call only.
        from scout.scout_controller import ScoutController
        tmp = ScoutController(interface=SCOUT_CAN_INTERFACE, dry_run=True)
        try:
            result = tmp.move_forward(
                distance_m=distance_m,
                speed_mps=speed_mps,
                timeout_s=timeout_s,
            )
        finally:
            tmp.close()
    else:
        sc = ensure_scout()
        result = sc.move_forward(
            distance_m=distance_m,
            speed_mps=speed_mps,
            timeout_s=timeout_s,
        )
    state.last_scout_move = result
    return result


def scout_move_backward(
    distance_m: float,
    speed_mps: float,
    timeout_s: float = None,
    dry_run: bool = False,
) -> dict:
    """Command a backward move. Updates last_scout_move."""
    if dry_run:
        # Override the controller's dry_run flag for this call only.
        from scout.scout_controller import ScoutController
        tmp = ScoutController(interface=SCOUT_CAN_INTERFACE, dry_run=True)
        try:
            result = tmp.move_backward(
                distance_m=distance_m,
                speed_mps=speed_mps,
                timeout_s=timeout_s,
            )
        finally:
            tmp.close()
    else:
        sc = ensure_scout()
        result = sc.move_backward(
            distance_m=distance_m,
            speed_mps=speed_mps,
            timeout_s=timeout_s,
        )
    state.last_scout_move = result
    return result


def scout_status_payload() -> dict:
    """Return a status dict for the /scout page."""
    return {
        "scout_enabled": SCOUT_ENABLED,
        "interface": SCOUT_CAN_INTERFACE,
        "dry_run_default": SCOUT_DRY_RUN,
        "advance_distance_m": SCOUT_ADVANCE_DISTANCE_M,
        "advance_speed_mps": SCOUT_ADVANCE_SPEED_MPS,
        "advance_timeout_sec": SCOUT_ADVANCE_TIMEOUT_SEC,
        "connected": state.scout is not None,
        "last_check": state.last_scout_check,
        "last_move": state.last_scout_move,
    }
