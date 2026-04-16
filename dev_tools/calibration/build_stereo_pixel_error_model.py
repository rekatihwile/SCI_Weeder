import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.pixel_error_model import fit_stereo_pixel_linear_correction


SAMPLES_PATH = ROOT / "planning" / "triangulation_calibration" / "manual_triangulation_samples.json"
MODEL_PATH = ROOT / "params" / "calibration" / "stereo_pixel_error_model.json"
SUMMARY_PATH = ROOT / "planning" / "triangulation_calibration" / "manual_triangulation_model_summary.json"


def load_samples(path):
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    samples = load_samples(SAMPLES_PATH)
    model, metrics = fit_stereo_pixel_linear_correction(samples)

    residual_preview = []
    for s in samples:
        pred_dx, pred_dy = model.predict_error(s["survey_left_px"], s["survey_right_px"])
        residual_preview.append({
            "sample_id": int(s["sample_id"]),
            "measured_error_xy_mm": [float(s["error_xy_mm"][0]), float(s["error_xy_mm"][1])],
            "predicted_error_xy_mm": [float(pred_dx), float(pred_dy)],
            "prediction_residual_xy_mm": [
                float(s["error_xy_mm"][0] - pred_dx),
                float(s["error_xy_mm"][1] - pred_dy),
            ],
        })

    model.save(
        MODEL_PATH,
        extra={
            "fit_metrics": metrics,
            "residual_preview": residual_preview,
            "source_samples_path": str(SAMPLES_PATH),
        },
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(MODEL_PATH),
                "fit_metrics": metrics,
                "basis_names": model.basis_names,
                "coeffs_x": model.coeffs_x.tolist(),
                "coeffs_y": model.coeffs_y.tolist(),
            },
            f,
            indent=2,
        )

    print("\n=== STEREO PIXEL ERROR MODEL ===")
    print(f"samples      : {metrics['n_samples']}")
    print(f"rmse_x_mm    : {metrics['rmse_x_mm']:.4f}")
    print(f"rmse_y_mm    : {metrics['rmse_y_mm']:.4f}")
    print(f"rmse_xy_mm   : {metrics['rmse_xy_mm']:.4f}")
    print(f"model saved  : {MODEL_PATH}")
    print(f"summary saved: {SUMMARY_PATH}")

    print("\nPaste into config.py if you want automatic correction later:")
    print('USE_PIXEL_ERROR_CORRECTION = True')
    print('PIXEL_ERROR_MODEL_PATH = BASE_DIR / "params/calibration/stereo_pixel_error_model.json"')


if __name__ == "__main__":
    main()