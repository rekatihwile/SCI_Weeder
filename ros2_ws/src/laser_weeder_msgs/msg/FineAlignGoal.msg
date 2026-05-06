# Send to /lw/fine_align/goal to start or cancel a fine-align session.
# Initial detection pixel coordinates are FULL-frame (pre-crop).
float32 left_x_px      # initial left-camera detection x (full frame)
float32 left_y_px      # initial left-camera detection y (full frame)
float32 right_x_px     # initial right-camera detection x (full frame)
float32 right_y_px     # initial right-camera detection y (full frame)
float32 deadzone_px    # 0 = use config FINE_ALIGN_DEADZONE_PX
float32 max_time_sec   # 0 = use config FINE_ALIGN_MAX_TIME_SEC
int32   settle_frames  # 0 = use config FINE_ALIGN_SETTLE_FRAMES
bool    cancel         # true = abort current session immediately
