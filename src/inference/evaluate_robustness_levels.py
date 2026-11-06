
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor

VALIDATION_CSV = PROJECT_ROOT / "data/processed/validation.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports/results"

SAMPLES_PER_CLASS = 5
SEED = 42

# Each condition has progressively stronger severity.
LEVELS = {
    "original": [None],
    "blur": [3, 5, 9],
    "low_light": [0.75, 0.55, 0.40],
    "low_resolution": [0.75, 0.50, 0.25],
    "jpeg": [70, 40, 25],
}


def transform(image, condition, severity, seed):
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]

    if condition == "original":
        return Image.fromarray(array)

    if condition == "blur":
        kernel = np.zeros((severity, severity), dtype=np.float32)
        kernel[severity // 2, :] = 1.0 / severity
        result = cv2.filter2D(array, -1, kernel)

    elif condition == "low_light":
        rng = np.random.default_rng(seed)
        dark = array.astype(np.float32) * severity
        noise = rng.normal(0, 5 + (0.40 - severity) * 12, array.shape)
        result = np.clip(dark + noise, 0, 255).astype(np.uint8)

    elif condition == "low_resolution":
        small = cv2.resize(
            array,
            (max(32, int(width * severity)),
             max(32, int(height * severity))),
            interpolation=cv2.INTER_AREA
        )
        result = cv2.resize(
            small, (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    elif condition == "jpeg":
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, severity]
        )
        if not success:
            raise RuntimeError("JPEG encoding failed")

        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(condition)

    return Image.fromarray(result)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(VALIDATION_CSV)

    sample = (
        df.groupby("label", group_keys=False)
        .sample(n=SAMPLES_PER_CLASS, random_state=SEED)
        .sort_values(["label", "image_path"])
        .reset_index(drop=True)
    )

    print("Images:", len(sample))
    print("Classes:", sample["label"].nunique())

    predictor = PlantDiseasePredictor()
    results = []

    total = sum(len(values) for values in LEVELS.values()) * len(sample)
    completed = 0

    for condition, severities in LEVELS.items():
        for severity in severities:
            level_name = (
                "original" if severity is None
                else f"{condition}_{severity}"
            )

            print(f"\nTesting: {level_name}")

            y_true = []
            y_pred = []
            confidences = []

            for index, row in sample.iterrows():
                path = PROJECT_ROOT / row["image_path"]

                with Image.open(path) as source:
                    original = source.convert("RGB")

                image = transform(
                    original, condition, severity,
                    SEED + index
                )

                prediction = predictor.predict_image(image)

                actual = row["label"]
                predicted = prediction["predicted_class"]
                confidence = prediction["confidence"]

                y_true.append(actual)
                y_pred.append(predicted)
                confidences.append(confidence)

                results.append({
                    "condition": condition,
                    "severity": severity,
                    "level": level_name,
                    "image_path": row["image_path"],
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "confidence": confidence,
                    "correct": actual == predicted
                })

                completed += 1

                if completed % 50 == 0:
                    print(f"Progress: {completed}/{total}")

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Mean confidence: {np.mean(confidences):.4f}")

    results_df = pd.DataFrame(results)

    predictions_path = OUTPUT_DIR / "robustness_levels_predictions.csv"
    summary_path = OUTPUT_DIR / "robustness_levels_summary.csv"

    results_df.to_csv(predictions_path, index=False)

    summary = (
        results_df.groupby(["condition", "level", "severity"], dropna=False)
        .agg(
            images=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean")
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(predictions_path)
    print(summary_path)


if __name__ == "__main__":
    main()
