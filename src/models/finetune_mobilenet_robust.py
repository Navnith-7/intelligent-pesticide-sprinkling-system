"""
Robust fine-tuning experiment for MobileNetV2.

Starting weights : models/best_mobilenet_v2_finetuned.pth
Training data    : data/processed/train.csv  (robust augmentation pipeline)
Validation data  : data/processed/validation.csv  (standard eval transforms)
Output checkpoint: models/best_mobilenet_v2_robust.pth
Training history : reports/results/mobilenet_v2_robust_history.csv

The original checkpoint is never touched by this script.
The test split is never loaded or evaluated here.

Run from the project root:
    python -m src.models.finetune_mobilenet_robust
"""

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import PlantVillageDataset, EVAL_TRANSFORM
from src.data.robust_transforms import ROBUST_TRAIN_TRANSFORM
from src.models.mobilenet import PlantDiseaseMobileNetV2


# ---------------------------------------------------------------------------
# Thin helper so callers can use the same function-call style as the task
# spec requests, without modifying robust_transforms.py.
# ---------------------------------------------------------------------------
def get_robust_train_transforms():
    """Return the shared ROBUST_TRAIN_TRANSFORM pipeline."""
    return ROBUST_TRAIN_TRANSFORM


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"

# Starting weights — protected, never overwritten by this script.
SOURCE_CHECKPOINT = MODEL_DIR / "best_mobilenet_v2_finetuned.pth"

# New checkpoint for this experiment only.
CHECKPOINT_PATH = MODEL_DIR / "best_mobilenet_v2_robust.pth"
HISTORY_PATH = RESULTS_DIR / "mobilenet_v2_robust_history.csv"

# ---------------------------------------------------------------------------
# Hyperparameters (fixed by task spec)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 10
LEARNING_RATE = 1e-5       # single uniform lr for all parameters
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0            # keep at 0 for Windows compatibility


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Pin all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, float]:
    """
    Run one full pass over `loader`.

    If `optimizer` is supplied the model is put in train mode and weights
    are updated; otherwise the model runs in eval mode (validation pass).

    Returns
    -------
    (mean_loss, accuracy, macro_f1)
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(predictions.detach().cpu().numpy().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return mean_loss, accuracy, macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guard: refuse to run if the source checkpoint is missing.
    # ------------------------------------------------------------------
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {SOURCE_CHECKPOINT}\n"
            "Run finetune_mobilenet.py first to produce it."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device                : {device}")
    print(f"Source checkpoint     : {SOURCE_CHECKPOINT}")
    print(f"Output checkpoint     : {CHECKPOINT_PATH}")
    print(f"History CSV           : {HISTORY_PATH}")
    print()

    # ------------------------------------------------------------------
    # Load class mapping from the source checkpoint so the label space
    # is exactly the same as the original training run.
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        SOURCE_CHECKPOINT, map_location=device, weights_only=False
    )
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class: dict[int, str] = checkpoint["idx_to_class"]
    num_classes = len(class_to_idx)

    if num_classes != 38:
        raise ValueError(
            f"Expected 38 classes but checkpoint contains {num_classes}. "
            "Verify the source checkpoint."
        )

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    robust_train_transform = get_robust_train_transforms()

    train_dataset = PlantVillageDataset(
        TRAIN_CSV,
        class_to_idx=class_to_idx,
        transform=robust_train_transform,
    )
    val_dataset = PlantVillageDataset(
        VAL_CSV,
        class_to_idx=class_to_idx,
        transform=EVAL_TRANSFORM,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training images       : {len(train_dataset):,}")
    print(f"Validation images     : {len(val_dataset):,}")
    print(f"Num classes           : {num_classes}")
    print()

    # ------------------------------------------------------------------
    # Model — initialise architecture then load source weights.
    # pretrained=False prevents downloading ImageNet weights; we
    # immediately overwrite everything with the checkpoint anyway.
    # ------------------------------------------------------------------
    model = PlantDiseaseMobileNetV2(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded weights from source checkpoint.")

    # Unfreeze ALL parameters so the entire network fine-tunes at lr=1e-5.
    for parameter in model.parameters():
        parameter.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters  : {trainable_count:,}")
    print()

    # ------------------------------------------------------------------
    # Optimiser, loss, and AMP scaler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # Training loop — 10 epochs, best checkpoint selected by val macro-F1
    # ------------------------------------------------------------------
    print(f"{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  {'TrF1':>6}  "
          f"{'ValLoss':>7}  {'ValAcc':>6}  {'ValF1':>6}  {'Secs':>6}  Best")
    print("-" * 80)

    best_val_f1 = -1.0
    best_epoch = 0
    history: list[dict] = []
    total_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device,
        )

        elapsed = time.time() - epoch_start
        is_best = val_f1 > best_val_f1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "train_macro_f1": round(train_f1, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "epoch_seconds": round(elapsed, 2),
            "is_best": is_best,
        })

        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "validation_accuracy": val_acc,
                    "validation_macro_f1": val_f1,
                    "validation_loss": val_loss,
                    "image_size": 224,
                    "normalization_mean": (0.485, 0.456, 0.406),
                    "normalization_std": (0.229, 0.224, 0.225),
                    "seed": SEED,
                    "architecture": "MobileNetV2_robust_finetuned",
                    "source_checkpoint": str(SOURCE_CHECKPOINT),
                    "augmentation": "ROBUST_TRAIN_TRANSFORM",
                    "optimizer": "AdamW",
                    "lr": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                },
                CHECKPOINT_PATH,
            )

        print(
            f"{epoch:>5d}  "
            f"{train_loss:>7.4f}  {train_acc:>6.4f}  {train_f1:>6.4f}  "
            f"{val_loss:>7.4f}  {val_acc:>6.4f}  {val_f1:>6.4f}  "
            f"{elapsed:>6.1f}s  {'*' if is_best else ''}"
        )

        # Save history after every epoch so progress is not lost on crash.
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_minutes = (time.time() - total_start) / 60
    print()
    print("=" * 60)
    print("ROBUST FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best val macro-F1     : {best_val_f1:.4f}")
    print(f"Total time            : {total_minutes:.2f} minutes")
    print(f"Checkpoint saved to   : {CHECKPOINT_PATH}")
    print(f"History saved to      : {HISTORY_PATH}")
    print()
    print("Protected checkpoints NOT modified:")
    print(f"  {MODEL_DIR / 'best_mobilenet_v2_finetuned.pth'}")
    print(f"  {MODEL_DIR / 'best_mobilenet_v2.pth'}")
    print(f"  {MODEL_DIR / 'best_cnn.pth'}")
    print("The test split was never loaded or evaluated.")


if __name__ == "__main__":
    main()

