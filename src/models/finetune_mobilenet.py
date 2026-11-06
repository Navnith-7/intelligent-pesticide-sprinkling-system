
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import (
    PlantVillageDataset,
    TRAIN_TRANSFORM,
    EVAL_TRANSFORM,
)
from src.models.mobilenet import PlantDiseaseMobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
BASELINE_PATH = MODEL_DIR / "best_mobilenet_v2.pth"
CHECKPOINT_PATH = MODEL_DIR / "best_mobilenet_v2_finetuned.pth"
HISTORY_PATH = RESULTS_DIR / "mobilenet_v2_finetuning_history.csv"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 12
PATIENCE = 4
HEAD_LR = 1e-4
FEATURE_LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

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

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(predictions.detach().cpu().numpy().tolist())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro", zero_division=0),
    )


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {BASELINE_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(
        BASELINE_PATH, map_location=device, weights_only=False
    )
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]
    num_classes = len(class_to_idx)

    train_dataset = PlantVillageDataset(
        TRAIN_CSV,
        class_to_idx=class_to_idx,
        transform=TRAIN_TRANSFORM,
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

    model = PlantDiseaseMobileNetV2(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze all feature blocks first.
    for parameter in model.model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune only the last two feature blocks.
    for block in model.model.features[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Keep the complete classifier trainable.
    for parameter in model.model.classifier.parameters():
        parameter.requires_grad = True

    feature_parameters = [
        p for p in model.model.features.parameters() if p.requires_grad
    ]
    head_parameters = list(model.model.classifier.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": feature_parameters, "lr": FEATURE_LR},
            {"params": head_parameters, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Training images: {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Feature learning rate: {FEATURE_LR}")
    print(f"Classifier learning rate: {HEAD_LR}")
    print("\nStarting fine-tuning...\n")

    best_val_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device,
        )

        seconds = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_seconds": seconds,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_count = 0

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
                    "architecture": "MobileNetV2_finetuned",
                },
                CHECKPOINT_PATH,
            )
        else:
            patience_count += 1

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"{seconds:.1f}s"
        )

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print("\n========== FINE-TUNING COMPLETE ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")
    print("The test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
