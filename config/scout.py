"""
config/scout.py — AgileX Scout mobile base integration.

KNOBS:
  SCOUT_ENABLED            — master enable; set True to allow Scout commands
  SCOUT_MOVE_AFTER_TRIAL   — move Scout forward after each successful trial
  SCOUT_REQUIRED_FOR_AUTO  — abort multi-trial run if Scout preflight fails
"""

# =============================================================================
# Enable flags
# =============================================================================

SCOUT_ENABLED = True
# Master enable. Must be True for any Scout motion to occur.

SCOUT_MOVE_AFTER_TRIAL = True
# If True (and SCOUT_ENABLED), move Scout forward after each completed trial.

SCOUT_REQUIRED_FOR_AUTO = False
# If True, abort multi-trial run if Scout preflight check fails.
# If False, warn and skip Scout movement if not connected.

# =============================================================================
# CAN / hardware
# =============================================================================

SCOUT_CAN_INTERFACE = "can0"
# SocketCAN interface name. Verify with: ip -details link show can0

# =============================================================================
# Motion parameters
# =============================================================================

SCOUT_ADVANCE_DISTANCE_M = 0.5
# How far to advance Scout forward after each trial, in meters.

SCOUT_ADVANCE_SPEED_MPS = 0.075
# Linear speed during advance, in m/s. Keep low for safety.

SCOUT_ADVANCE_TIMEOUT_SEC = SCOUT_ADVANCE_DISTANCE_M / SCOUT_ADVANCE_SPEED_MPS * 3.0
# Abort Scout move if it takes longer than this, in seconds.

SCOUT_SETTLE_SEC =3.0
# Pause this long after a Scout move before starting the next trial, in seconds.

# =============================================================================
# Debug
# =============================================================================

SCOUT_DRY_RUN = False
# If True, print intended commands but do not send any CAN motion.
