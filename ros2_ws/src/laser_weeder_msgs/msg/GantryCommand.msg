# cmd: "move_abs" | "jog" | "home" | "fire" | "stop" | "unlock"
string  cmd
float32 x_mm          # move_abs / jog target or delta X
float32 y_mm          # move_abs / jog target or delta Y
float32 feed_rate     # mm/min; 0 = use config default
float32 duration_sec  # fire pulse duration; 0 = use config default
