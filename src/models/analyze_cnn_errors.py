
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

report_path = RESULTS_DIR / "cnn_test_classification_report.csv"
matrix_path = RESULTS_DIR / "cnn_test_confusion_matrix.csv"
predictions_path = RESULTS_DIR / "cnn_test_predictions.csv"

report = pd.read_csv(report_path, index_col=0)
matrix = pd.read_csv(matrix_path, index_col=0)
predictions = pd.read_csv(predictions_path)

# Keep only individual class rows, excluding aggregate metric rows.
class_report = report.loc[
    ~report.index.isin(["accuracy", "macro avg", "weighted avg"])
].copy()

class_report = class_report[
    pd.to_numeric(class_report["support"], errors="coerce").notna()
]

class_report["support"] = class_report["support"].astype(int)
class_report = class_report.sort_values("f1-score")

print("\n========== 10 WEAKEST CLASSES ==========")
print(
    class_report[["precision", "recall", "f1-score", "support"]]
    .head(10)
    .round(4)
    .to_string()
)

# Find the largest off-diagonal confusion counts.
confusions = []

for true_label in matrix.index:
    for predicted_label in matrix.columns:
        if true_label != predicted_label:
            count = int(matrix.loc[true_label, predicted_label])
            if count > 0:
                confusions.append(
                    (true_label, predicted_label, count)
                )

confusions.sort(key=lambda item: item[2], reverse=True)

print("\n========== TOP 15 CLASS CONFUSIONS ==========")
print(f"{'ACTUAL CLASS':40} {'PREDICTED CLASS':40} {'COUNT':>6}")
for true_label, predicted_label, count in confusions[:15]:
    print(f"{true_label[:40]:40} {predicted_label[:40]:40} {count:6d}")

# Summarize individual errors.
errors = predictions[~predictions["correct"].astype(bool)]

print("\n========== ERROR SUMMARY ==========")
print(f"Incorrect predictions: {len(errors):,}")
print(f"Error rate: {len(errors) / len(predictions):.2%}")

print("\nTop actual classes contributing errors:")
print(errors["true_label"].value_counts().head(10).to_string())

print("\nAnalysis complete. No model weights or test data were modified.")
