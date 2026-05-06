# Single object detection on one camera side.
float32[4] box       # x1 y1 x2 y2 in full-frame pixels
float32[2] point     # cx cy centre in full-frame pixels
int32      cls
float32    conf
int32      views     # burst frames that agreed (1 for live single-frame)
