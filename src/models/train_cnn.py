
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import (
    PlantVillageDataset,
    TRAIN_TRANSFORM,
    EVAL_TRANSFORM,
)
from src.models.cnn import PlantDiseaseCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "validation.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_WORKERS = 0
SEED = 42

MODEL_PATH = MODEL_DIR / "best_cnn.pth"
HISTORY_PATH = RESULTS_DIR / "cnn_training_history.csv"


# --------------------------------------------------
# Reproducibility and device
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# One training epoch
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    use_amp = device.type == "cuda"

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# Validation epoch
# --------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    use_amp = device.type == "cuda"

    with torch.inference_mode():
        progress = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# Main training workflow
# --------------------------------------------------

def main():
    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 60)
    print("PLANT DISEASE CNN TRAINING")
    print("=" * 60)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    print("\nLoading datasets...")

    train_dataset = PlantVillageDataset(
        TRAIN_CSV,
        transform=TRAIN_TRANSFORM,
    )

    val_dataset = PlantVillageDataset(
        VAL_CSV,
        class_to_idx=train_dataset.class_to_idx,
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

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {
        index: label for label, index in class_to_idx.items()
    }

    model = PlantDiseaseCNN(
        num_classes=len(class_to_idx)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(device.type == "cuda"),
    )

    print("Training images:", f"{len(train_dataset):,}")
    print("Validation images:", f"{len(val_dataset):,}")
    print("Number of classes:", len(class_to_idx))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", NUM_EPOCHS)
    print("Mixed precision:", device.type == "cuda")
    print("\nStarting training...\n")

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(
            f"Train | Loss: {train_loss:.4f} | "
            f"Accuracy: {train_acc:.4f} | Macro-F1: {train_f1:.4f}"
        )
        print(
            f"Val   | Loss: {val_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}"
        )
        print(f"Epoch time: {epoch_seconds / 60:.2f} minutes")

        # Save a checkpoint only when validation macro-F1 improves.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "image_size": 224,
                    "normalization_mean": (0.485, 0.456, 0.406),
                    "normalization_std": (0.229, 0.224, 0.225),
                    "seed": SEED,
                },
                MODEL_PATH,
            )

            print("New best checkpoint saved:", MODEL_PATH)
        else:
            epochs_without_improvement += 1
            print(
                "No validation macro-F1 improvement:",
                f"{epochs_without_improvement}/{PATIENCE}",
            )

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    elapsed_minutes = (time.time() - start_time) / 60

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print("Best epoch:", best_epoch)
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Total training time: {elapsed_minutes:.2f} minutes")
    print("Best checkpoint:", MODEL_PATH)
    print("Training history:", HISTORY_PATH)
    print("\nThe test split has NOT been evaluated.")


if __name__ == "__main__":
    main()
