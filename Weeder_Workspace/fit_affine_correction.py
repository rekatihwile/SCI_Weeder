from control.calibration_correction import (
    fit_affine_xy_correction,
    print_affine_correction,
)


samples = [
    {"est": (244.76,148.99), "true": (250.0, 100.0)}, 
    {"est": (151.08,98.47), "true": (150.0, 100.0)},
    {"est": (100,97.75), "true": (100.0, 100.0)},
    {"est": (97.12,198.48), "true": (100.0, 200.0)},
    {"est": (148.09,198.74), "true": (150.0, 200.0)},
    {"est": (-6.418,198.69), "true": (0.0, 200.0)},
    {"est": (52.46,196.25), "true": (50.0, 200.0)},
    {"est": (50.457,246.818), "true": (50.0, 250.0)},
    {"est": (200.71, 149.11), "true": (200.0, 150.0)}, #
    {"est": (99.97, 147.46), "true": (100.0, 150.0)}, #
    {"est": (248.98,200.22), "true": (250.0, 200.0)}, #
    {"est": (150.22,148.47), "true": (150.0, 150.0)} #
]

correction = fit_affine_xy_correction(samples)
print_affine_correction(correction)

print("\nPaste these into config.py:")
print(f"AFFINE_X_COEFFS = {correction.x_coeffs.tolist()}")
print(f"AFFINE_Y_COEFFS = {correction.y_coeffs.tolist()}")