import json
import numpy as np

class StereoPixelErrorModel:
    def __init__(self, model_path=None):
        self.coeffs_x = None
        self.coeffs_y = None
        self.basis_names = ["1", "xl", "yl", "xr", "yr", "xl-xr", "yl-yr"]
        self.enabled = False
        
        if model_path:
            self.load(model_path)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.coeffs_x = np.array(data['coeffs_x'])
        self.coeffs_y = np.array(data['coeffs_y'])
        self.enabled = True

    def save(self, path, extra=None):
        out = {
            "coeffs_x": self.coeffs_x.tolist(),
            "coeffs_y": self.coeffs_y.tolist(),
            "basis_names": self.basis_names
        }
        if extra:
            out.update(extra)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    def predict_error(self, left_px, right_px):
        """Returns (dx, dy) in mm to be ADDED to the raw triangulation."""
        if not self.enabled:
            return 0.0, 0.0
            
        xl, yl = left_px
        xr, yr = right_px
        
        # 7-term basis to match the training function below
        basis = np.array([
            1, 
            xl, 
            yl, 
            xr, 
            yr, 
            xl - xr, 
            yl - yr
        ])
        
        dx = float(np.dot(basis, self.coeffs_x))
        dy = float(np.dot(basis, self.coeffs_y))
        return dx, dy

def fit_stereo_pixel_linear_correction(samples):
    """
    Fits a linear model to map survey pixel coordinates to triangulation error.
    Basis: [1, xl, yl, xr, yr, (xl-xr), (yl-yr)]
    """
    if len(samples) < 8:
        print(f"[WARNING] Only {len(samples)} samples found. Linear model may be overfit.")

    A = []
    bx = []
    by = []

    for s in samples:
        xl, yl = s["survey_left_px"]
        xr, yr = s["survey_right_px"]
        ex, ey = s["error_xy_mm"]

        # Build the 7-term row
        A.append([
            1.0, 
            xl, 
            yl, 
            xr, 
            yr, 
            xl - xr, 
            yl - yr
        ])
        bx.append(ex)
        by.append(ey)

    A = np.array(A)
    bx = np.array(bx)
    by = np.array(by)

    # Solve for coefficients using Least Squares
    coeffs_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Calculate Metrics
    pred_x = A @ coeffs_x
    pred_y = A @ coeffs_y
    res_x = bx - pred_x
    res_y = by - pred_y
    
    rmse_x = np.sqrt(np.mean(res_x**2))
    rmse_y = np.sqrt(np.mean(res_y**2))

    model = StereoPixelErrorModel()
    model.coeffs_x = coeffs_x
    model.coeffs_y = coeffs_y
    model.enabled = True

    metrics = {
        "n_samples": len(samples),
        "rmse_x_mm": float(rmse_x),
        "rmse_y_mm": float(rmse_y),
        "rmse_xy_mm": float(np.sqrt(rmse_x**2 + rmse_y**2))
    }

    return model, metrics