std_msgs/Header header
float32 dx_px        # error X: (xl+xr) - 2*centre_x  (positive = plant right of laser)
float32 dy_px        # error Y: centre_y - (yl+yr)/2   (positive = plant below laser)
bool    in_deadzone  # |dx|<=deadzone and |dy|<=deadzone
bool    locked       # settled for the required consecutive frames
bool    timed_out
float32 elapsed_sec
