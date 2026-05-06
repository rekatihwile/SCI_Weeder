# A stereo-matched + triangulated target in gantry workspace coordinates.
float32[2] left_px    # detection centre in left camera full-frame pixels
float32[2] right_px   # detection centre in right camera full-frame pixels
float32    x_mm       # gantry workspace X (absolute, mm)
float32    y_mm       # gantry workspace Y (absolute, mm)
float32    score      # stereo match score
int32      cls
float32    conf
