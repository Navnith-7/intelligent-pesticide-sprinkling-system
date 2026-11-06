
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.data.dataset import PlantVillageDataset, EVAL_TRANSFORM
from src.models.cnn import PlantDiseaseCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_cnn.pth"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

BATCH_SIZE = 32


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the saved checkpoint and its original class mapping.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        int(index): label
        for index, label in checkpoint["idx_to_class"].items()
    }
    num_classes = len(class_to_idx)

    model = PlantDiseaseCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load only the held-out test split. Do not augment test images.
    test_dataset = PlantVillageDataset(
        csv_path=TEST_CSV,
        class_to_idx=class_to_idx,
        transform=EVAL_TRANSFORM,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test images: {len(test_dataset):,}")
    print(f"Number of classes: {num_classes}")
    print("\nEvaluating the saved model...")

    all_true = []
    all_pred = []
    all_confidence = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)

            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(predictions.cpu().numpy().tolist())
            all_confidence.extend(confidence.cpu().numpy().tolist())

    # Calculate overall test metrics.
    accuracy = accuracy_score(all_true, all_pred)
    macro_precision = precision_score(
        all_true, all_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        all_true, all_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        all_true, all_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        all_true, all_pred, average="weighted", zero_division=0
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test accuracy:       {accuracy:.4%}")
    print(f"Macro precision:     {macro_precision:.4f}")
    print(f"Macro recall:        {macro_recall:.4f}")
    print(f"Macro F1-score:      {macro_f1:.4f}")
    print(f"Weighted F1-score:   {weighted_f1:.4f}")
    print("==================================")

    # Detailed metrics for every class.
    class_names = [idx_to_class[i] for i in range(num_classes)]

    report_dict = classification_report(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = RESULTS_DIR / "cnn_test_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    # Save confusion matrix with class names on both axes.
    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_path = RESULTS_DIR / "cnn_test_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path, index=True)

    # Save individual predictions for later error analysis.
    test_rows = test_dataset.data.reset_index(drop=True).copy()
    test_rows["true_label"] = [
        idx_to_class[index] for index in all_true
    ]
    test_rows["predicted_label"] = [
        idx_to_class[index] for index in all_pred
    ]
    test_rows["confidence"] = all_confidence
    test_rows["correct"] = [
        true == pred for true, pred in zip(all_true, all_pred)
    ]

    predictions_path = RESULTS_DIR / "cnn_test_predictions.csv"
    test_rows.to_csv(predictions_path, index=False)

    print("\nSaved evaluation files:")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix:      {matrix_path}")
    print(f"Predictions:           {predictions_path}")

    print("\nSUCCESS: test evaluation completed.")
    print("Keep this test set untouched for model selection and threshold tuning.")


if __name__ == "__main__":
    main()
