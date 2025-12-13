class PixelPDController:
    def __init__(self):
        self.prev_eX = 0.0
        self.prev_eY = 0.0

        self.tol_x = 3
        self.tol_y = 3
        self.step_mm = 0.001

        self.K_Px = 5
        self.K_Dx = 5
        self.K_Py = 11
        self.K_Dy = 3

    def step(self, frame_width, xL, yL, xR, yR):
        eX = horizontal_error(xL, xR, frame_width)
        deX = eX - self.prev_eX
        self.prev_eX = eX

        if abs(eX) > self.tol_x and abs(deX*self.K_Dx) < abs(eX*self.K_Px):
            dx = (eX*self.K_Px + deX*self.K_Dx)*self.step_mm
        else:
            dx = 0.0

        eYL = vertical_error(xL, yL, mL, bL)
        eYR = vertical_error(xR, yR, mR, bR)
        eY = 0.5*(eYL + eYR)

        deY = eY - self.prev_eY
        self.prev_eY = eY

        if abs(eY) > self.tol_y and abs(deY*self.K_Dy) < abs(eY*self.K_Py):
            dy = (eY*self.K_Py + deY*self.K_Dy)*self.step_mm
        else:
            dy = 0.0

        aligned = abs(eX) <= self.tol_x and abs(eY) <= self.tol_y
        return round(dx, 3), round(dy, 3), aligned


# Vertical error function
def vertical_error(x, y, m, b):
    return  (m * x + b) - y

# Horizontal symmetry error function
def horizontal_error(xL, xR, frame_width):
    dL = frame_width - xL   # dist from right edge (left cam)
    dR = xR                 # dist from left edge  (right cam)
    return dR - dL

