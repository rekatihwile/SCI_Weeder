import numpy as np


class AffineXYCorrection:
    def __init__(self, x_coeffs, y_coeffs):
        self.x_coeffs = np.array(x_coeffs, dtype=float)  # [a0, a1, a2]
        self.y_coeffs = np.array(y_coeffs, dtype=float)  # [b0, b1, b2]

    def apply(self, x_est, y_est):
        v = np.array([1.0, x_est, y_est], dtype=float)
        x_corr = float(self.x_coeffs @ v)
        y_corr = float(self.y_coeffs @ v)
        return x_corr, y_corr


def fit_affine_xy_correction(samples):
    """
    samples: list of dicts like
    {
        "est": (x_est, y_est),
        "true": (x_true, y_true),
    }
    """
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples for affine correction.")

    A = []
    bx = []
    by = []

    for s in samples:
        x_est, y_est = s["est"]
        x_true, y_true = s["true"]

        A.append([1.0, x_est, y_est])
        bx.append(x_true)
        by.append(y_true)

    A = np.array(A, dtype=float)
    bx = np.array(bx, dtype=float)
    by = np.array(by, dtype=float)

    x_coeffs, *_ = np.linalg.lstsq(A, bx, rcond=None)
    y_coeffs, *_ = np.linalg.lstsq(A, by, rcond=None)

    return AffineXYCorrection(x_coeffs, y_coeffs)


def print_affine_correction(correction):
    print("\n=== AFFINE CORRECTION ===")
    print(
        f"X_true = {correction.x_coeffs[0]:.6f}"
        f" + {correction.x_coeffs[1]:.6f}*X_est"
        f" + {correction.x_coeffs[2]:.6f}*Y_est"
    )
    print(
        f"Y_true = {correction.y_coeffs[0]:.6f}"
        f" + {correction.y_coeffs[1]:.6f}*X_est"
        f" + {correction.y_coeffs[2]:.6f}*Y_est"
    )