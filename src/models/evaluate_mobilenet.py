
import argparse
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
from src.models.mobilenet import PlantDiseaseMobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenet_v2_finetuned.pth"
DEFAULT_PREFIX = "mobilenet_v2_test"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
BATCH_SIZE = 32


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a MobileNetV2 checkpoint on the held-out test split."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Checkpoint to evaluate (default: models/best_mobilenet_v2_finetuned.pth).",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=(
            "Filename prefix for the three output CSVs in reports/results/ "
            "(default: mobilenet_v2_test). Use a new prefix to avoid overwriting."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    prefix = args.prefix

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_CSV}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {model_path}")
    print(f"Output prefix: {prefix}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        int(index): label
        for index, label in checkpoint["idx_to_class"].items()
    }
    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = PlantDiseaseMobileNetV2(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = PlantVillageDataset(
        TEST_CSV,
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

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test images: {len(test_dataset):,}")
    print(f"Classes: {num_classes}")
    print("\nEvaluating test set...")

    all_true = []
    all_pred = []
    all_confidence = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predictions = probabilities.max(dim=1)

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(predictions.cpu().numpy().tolist())
            all_confidence.extend(confidence.cpu().numpy().tolist())

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

    print("\n========== MOBILENETV2 TEST RESULTS ==========")
    print(f"Test accuracy:       {accuracy:.4%}")
    print(f"Macro precision:     {macro_precision:.4f}")
    print(f"Macro recall:        {macro_recall:.4f}")
    print(f"Macro F1-score:      {macro_f1:.4f}")
    print(f"Weighted F1-score:   {weighted_f1:.4f}")
    print("==============================================")
    correct_count = sum(int(t == p) for t, p in zip(all_true, all_pred))
    print(f"Correct / total:     {correct_count:,} / {len(all_true):,}")
    print(f"Errors:              {len(all_true) - correct_count:,}")

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / f"{prefix}_classification_report.csv"
    )

    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
    )

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(RESULTS_DIR / f"{prefix}_confusion_matrix.csv")

    predictions_df = test_dataset.data.reset_index(drop=True).copy()
    predictions_df["true_label"] = [
        idx_to_class[i] for i in all_true
    ]
    predictions_df["predicted_label"] = [
        idx_to_class[i] for i in all_pred
    ]
    predictions_df["confidence"] = all_confidence
    predictions_df["correct"] = [
        true == pred for true, pred in zip(all_true, all_pred)
    ]

    predictions_df.to_csv(
        RESULTS_DIR / f"{prefix}_predictions.csv",
        index=False,
    )

    print("\nSaved:")
    print(f" - {prefix}_classification_report.csv")
    print(f" - {prefix}_confusion_matrix.csv")
    print(f" - {prefix}_predictions.csv")
    print("\nSUCCESS: MobileNetV2 test evaluation completed.")


if __name__ == "__main__":
    main()
