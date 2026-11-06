
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import PlantVillageDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from src.models.mobilenet import PlantDiseaseMobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
CHECKPOINT_PATH = MODEL_DIR / "best_mobilenet_v2.pth"
HISTORY_PATH = RESULTS_DIR / "mobilenet_v2_training_history.csv"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-3
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

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise FileNotFoundError(
            "Training or validation CSV is missing. "
            "Run the dataset split script first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    classes = sorted(train_df["label"].unique())
    class_to_idx = {label: index for index, label in enumerate(classes)}
    idx_to_class = {index: label for label, index in class_to_idx.items()}

    if set(val_df["label"].unique()) != set(classes):
        raise ValueError("Training and validation classes do not match.")

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
        num_classes=len(classes),
        pretrained=True,
    ).to(device)

    # Train only the new classifier initially.
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    print(f"Training images: {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")
    print("\nStarting training...\n")

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
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

        epoch_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        improved = val_f1 > best_val_f1

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

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
                    "architecture": "MobileNetV2",
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    total_seconds = time.time() - start_time

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Total training time: {total_seconds / 60:.2f} minutes")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")
    print("The test set has not been loaded or evaluated.")


if __name__ == "__main__":
    main()
