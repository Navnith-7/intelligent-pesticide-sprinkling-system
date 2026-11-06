
from pathlib import Path
import sys
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor


# ---------------- CONFIGURATION ----------------

VALIDATION_CSV = PROJECT_ROOT / "data" / "processed" / "validation.csv"

SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "reports" / "results"

CONDITIONS = [
    "original",
    "motion_blur",
    "low_light_noise",
    "low_resolution",
    "jpeg_compression"
]


# ---------------- IMAGE VARIATIONS ----------------

def create_variation(image, condition, seed):

    image = image.convert("RGB")
    array = np.array(image)
    height, width = array.shape[:2]

    if condition == "original":
        return image

    elif condition == "motion_blur":
        kernel = np.zeros((9, 9), dtype=np.float32)
        kernel[4, :] = 1.0 / 9
        result = cv2.filter2D(array, -1, kernel)

    elif condition == "low_light_noise":
        rng = np.random.default_rng(seed)

        dark = array.astype(np.float32) * 0.40

        noise = rng.normal(
            0, 10, array.shape
        )

        result = np.clip(
            dark + noise, 0, 255
        ).astype(np.uint8)

    elif condition == "low_resolution":

        small = cv2.resize(
            array,
            (
                max(32, width // 4),
                max(32, height // 4)
            ),
            interpolation=cv2.INTER_AREA
        )

        result = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    elif condition == "jpeg_compression":

        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 25]
        )

        if not success:
            raise RuntimeError("JPEG encoding failed")

        decoded = cv2.imdecode(
            encoded, cv2.IMREAD_COLOR
        )

        result = cv2.cvtColor(
            decoded, cv2.COLOR_BGR2RGB
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return Image.fromarray(result)


# ---------------- MAIN EVALUATION ----------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(VALIDATION_CSV)

    df = pd.read_csv(VALIDATION_CSV)

    required_columns = {"image_path", "label", "sha256"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain {required_columns}"
        )

    print("Validation images:", len(df))

    class_counts = df["label"].value_counts()

    if len(class_counts) != 38:
        raise ValueError(
            f"Expected 38 classes, found {len(class_counts)}"
        )

    if (class_counts < SAMPLES_PER_CLASS).any():
        raise ValueError(
            "Some classes have fewer than five images."
        )

    # Deterministic, balanced sample.
    sample = (
        df.groupby("label", group_keys=False)
        .sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_SEED
        )
        .sort_values(["label", "image_path"])
        .reset_index(drop=True)
    )

    print("Classes:", sample["label"].nunique())
    print("Images selected:", len(sample))
    print("Conditions:", len(CONDITIONS))
    print("Total evaluations:", len(sample) * len(CONDITIONS))

    print("\nLoading MobileNetV2...")
    predictor = PlantDiseasePredictor()

    all_results = []

    for condition in CONDITIONS:

        print(f"\n========== {condition.upper()} ==========")

        condition_results = []

        for index, row in sample.iterrows():

            image_path = PROJECT_ROOT / row["image_path"]

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            actual_class = row["label"]

            with Image.open(image_path) as source:
                original = source.convert("RGB")

            image = create_variation(
                original,
                condition,
                seed=RANDOM_SEED + index
            )

            prediction = predictor.predict_image(image)

            predicted_class = prediction["predicted_class"]
            confidence = prediction["confidence"]

            correct = predicted_class == actual_class

            result = {
                "condition": condition,
                "image_path": row["image_path"],
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": correct
            }

            condition_results.append(result)
            all_results.append(result)

            if (index + 1) % 38 == 0:
                print(
                    f"Processed {index + 1}/{len(sample)} images"
                )

        y_true = [
            result["actual_class"]
            for result in condition_results
        ]

        y_pred = [
            result["predicted_class"]
            for result in condition_results
        ]

        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    # Save every prediction.
    predictions_df = pd.DataFrame(all_results)

    predictions_path = (
        OUTPUT_DIR / "robustness_190_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path, index=False
    )

    # Overall metrics.
    summary = []

    class_names = sorted(sample["label"].unique())

    for condition in CONDITIONS:

        subset = predictions_df[
            predictions_df["condition"] == condition
        ]

        y_true = subset["actual_class"]
        y_pred = subset["predicted_class"]

        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=class_names,
            average="macro",
            zero_division=0
        )

        summary.append({
            "condition": condition,
            "images": len(subset),
            "correct": int(subset["correct"].sum()),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mean_confidence": subset["confidence"].mean()
        })

        # Per-class performance.
        report = classification_report(
            y_true,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(
            OUTPUT_DIR / f"report_{condition}.csv"
        )

        # Confusion matrix.
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=class_names
        )

        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names
        ).to_csv(
            OUTPUT_DIR / f"confusion_{condition}.csv"
        )

    summary_df = pd.DataFrame(summary)

    summary_path = (
        OUTPUT_DIR / "robustness_190_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n========== FINAL RESULTS ==========")
    print(summary_df.to_string(index=False))

    print("\nFiles saved:")
    print(predictions_path)
    print(summary_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
