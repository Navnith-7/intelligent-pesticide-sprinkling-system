"""
Robustness evaluation for the robust-fine-tuned MobileNetV2 model.

Loads weights exclusively from:
    models/best_mobilenet_v2_robust.pth

Applies the identical 190-image balanced sample and the same four
degradation conditions used in evaluate_robustness.py so results are
directly comparable.

Outputs (all new files — nothing in evaluate_robustness.py is touched):
    reports/results/robust_model_190_predictions.csv
    reports/results/robust_model_190_summary.csv
    reports/results/robust_model_report_{condition}.csv
    reports/results/robust_model_confusion_{condition}.csv

Run from the project root:
    python -m src.inference.evaluate_robust_model
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when the script is run directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PlantDiseasePredictor accepts an optional model_path argument —
# we pass our robust checkpoint without touching predict.py at all.
from src.inference.predict import PlantDiseasePredictor


# ---------------------------------------------------------------------------
# Configuration — mirrors evaluate_robustness.py exactly.
# ---------------------------------------------------------------------------
VALIDATION_CSV = PROJECT_ROOT / "data" / "processed" / "validation.csv"
ROBUST_CHECKPOINT = PROJECT_ROOT / "models" / "best_mobilenet_v2_robust.pth"

SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "reports" / "results"

CONDITIONS = [
    "original",
    "motion_blur",
    "low_light_noise",
    "low_resolution",
    "jpeg_compression",
]


# ---------------------------------------------------------------------------
# Image degradation — copied verbatim from evaluate_robustness.py so the
# pixel-level transforms are byte-for-byte identical.
# ---------------------------------------------------------------------------
def create_variation(image: Image.Image, condition: str, seed: int) -> Image.Image:
    """Apply a deterministic degradation to a PIL RGB image."""

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
        noise = rng.normal(0, 10, array.shape)
        result = np.clip(dark + noise, 0, 255).astype(np.uint8)

    elif condition == "low_resolution":
        small = cv2.resize(
            array,
            (max(32, width // 4), max(32, height // 4)),
            interpolation=cv2.INTER_AREA,
        )
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    elif condition == "jpeg_compression":
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 25],
        )
        if not success:
            raise RuntimeError("JPEG encoding failed.")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Pretty terminal table
# ---------------------------------------------------------------------------
def _print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted comparison table to the terminal."""
    col_w = {
        "condition": 20,
        "images": 7,
        "correct": 8,
        "accuracy": 10,
        "macro_f1": 10,
        "mean_confidence": 16,
    }
    header = (
        f"{'Condition':<{col_w['condition']}} "
        f"{'Images':>{col_w['images']}} "
        f"{'Correct':>{col_w['correct']}} "
        f"{'Accuracy':>{col_w['accuracy']}} "
        f"{'Macro-F1':>{col_w['macro_f1']}} "
        f"{'Mean Conf':>{col_w['mean_confidence']}}"
    )
    separator = "-" * len(header)

    print()
    print("=" * len(header))
    print("  ROBUST MODEL — 190-IMAGE STRESS TEST RESULTS")
    print("=" * len(header))
    print(header)
    print(separator)

    for _, row in summary_df.iterrows():
        print(
            f"{row['condition']:<{col_w['condition']}} "
            f"{int(row['images']):>{col_w['images']}} "
            f"{int(row['correct']):>{col_w['correct']}} "
            f"{row['accuracy']:>{col_w['accuracy']}.4f} "
            f"{row['macro_f1']:>{col_w['macro_f1']}.4f} "
            f"{row['mean_confidence']:>{col_w['mean_confidence']}.4f}"
        )

    print(separator)
    print(
        f"Model checkpoint : {ROBUST_CHECKPOINT.name}\n"
        f"Sample           : {SAMPLES_PER_CLASS} images × 38 classes = "
        f"{SAMPLES_PER_CLASS * 38} images\n"
        f"Random seed      : {RANDOM_SEED}"
    )
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guard: refuse to run if the robust checkpoint is missing.
    # ------------------------------------------------------------------
    if not ROBUST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Robust checkpoint not found: {ROBUST_CHECKPOINT}\n"
            "Run finetune_mobilenet_robust.py first to produce it."
        )

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VALIDATION_CSV}")

    # ------------------------------------------------------------------
    # Load and validate the validation manifest.
    # ------------------------------------------------------------------
    df = pd.read_csv(VALIDATION_CSV)

    required_columns = {"image_path", "label", "sha256"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    print(f"Validation images total : {len(df):,}")

    class_counts = df["label"].value_counts()
    if len(class_counts) != 38:
        raise ValueError(f"Expected 38 classes, found {len(class_counts)}")

    if (class_counts < SAMPLES_PER_CLASS).any():
        raise ValueError("Some classes have fewer than 5 images in the validation set.")

    # ------------------------------------------------------------------
    # Deterministic, balanced 190-image sample — identical to the
    # original evaluate_robustness.py logic.
    # ------------------------------------------------------------------
    sample = (
        df.groupby("label", group_keys=False)
        .sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
        .sort_values(["label", "image_path"])
        .reset_index(drop=True)
    )

    print(f"Classes selected        : {sample['label'].nunique()}")
    print(f"Images selected         : {len(sample)}")
    print(f"Degradation conditions  : {len(CONDITIONS)}")
    print(f"Total evaluations       : {len(sample) * len(CONDITIONS)}")

    # ------------------------------------------------------------------
    # Load the robust model checkpoint.
    # PlantDiseasePredictor accepts model_path — predict.py is untouched.
    # ------------------------------------------------------------------
    print(f"\nLoading robust MobileNetV2 from: {ROBUST_CHECKPOINT.name}")
    predictor = PlantDiseasePredictor(model_path=ROBUST_CHECKPOINT)

    # ------------------------------------------------------------------
    # Evaluate each condition.
    # ------------------------------------------------------------------
    all_results: list[dict] = []

    for condition in CONDITIONS:
        print(f"\n========== {condition.upper()} ==========")

        condition_results: list[dict] = []

        for index, row in sample.iterrows():
            image_path = PROJECT_ROOT / row["image_path"]

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            actual_class = row["label"]

            with Image.open(image_path) as source:
                original = source.convert("RGB")

            # Apply the deterministic degradation.
            image = create_variation(
                original,
                condition,
                seed=RANDOM_SEED + index,
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
                "correct": correct,
            }

            condition_results.append(result)
            all_results.append(result)

            # Progress heartbeat (matches original cadence).
            if (index + 1) % 38 == 0:
                print(f"  Processed {index + 1}/{len(sample)} images")

        # Per-condition quick summary during the run.
        y_true_cond = [r["actual_class"] for r in condition_results]
        y_pred_cond = [r["predicted_class"] for r in condition_results]
        acc_cond = accuracy_score(y_true_cond, y_pred_cond)
        f1_cond = f1_score(y_true_cond, y_pred_cond, average="macro", zero_division=0)
        print(f"  Accuracy : {acc_cond:.4f}")
        print(f"  Macro-F1 : {f1_cond:.4f}")

    # ------------------------------------------------------------------
    # Persist all predictions.
    # ------------------------------------------------------------------
    predictions_df = pd.DataFrame(all_results)
    predictions_path = OUTPUT_DIR / "robust_model_190_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # ------------------------------------------------------------------
    # Compute summary + per-class artifacts for every condition.
    # ------------------------------------------------------------------
    summary: list[dict] = []
    class_names = sorted(sample["label"].unique())

    for condition in CONDITIONS:
        subset = predictions_df[predictions_df["condition"] == condition]

        y_true = subset["actual_class"]
        y_pred = subset["predicted_class"]

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )
        mean_confidence = subset["confidence"].mean()

        summary.append(
            {
                "condition": condition,
                "images": len(subset),
                "correct": int(subset["correct"].sum()),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "mean_confidence": mean_confidence,
            }
        )

        # Per-class classification report.
        report = classification_report(
            y_true,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / f"robust_model_report_{condition}.csv"
        )

        # Confusion matrix.
        matrix = confusion_matrix(y_true, y_pred, labels=class_names)
        pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
            OUTPUT_DIR / f"robust_model_confusion_{condition}.csv"
        )

    # ------------------------------------------------------------------
    # Save summary CSV.
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "robust_model_190_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # Print formatted comparison table.
    # ------------------------------------------------------------------
    _print_summary_table(summary_df)

    print("Files saved:")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    for condition in CONDITIONS:
        print(f"  {OUTPUT_DIR / f'robust_model_report_{condition}.csv'}")
        print(f"  {OUTPUT_DIR / f'robust_model_confusion_{condition}.csv'}")

    print("\nEvaluation completed successfully.")
    print("No existing checkpoints or scripts were modified.")


if __name__ == "__main__":
    main()

