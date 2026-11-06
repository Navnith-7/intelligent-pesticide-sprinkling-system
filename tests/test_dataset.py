
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    PlantVillageDataset,
    TRAIN_TRANSFORM,
    EVAL_TRANSFORM,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    # 1. Load the training dataset and establish the class mapping.
    train_dataset = PlantVillageDataset(
        PROCESSED_DIR / "train.csv",
        transform=TRAIN_TRANSFORM,
    )

    # 2. Use the same class mapping for validation and test.
    val_dataset = PlantVillageDataset(
        PROCESSED_DIR / "validation.csv",
        class_to_idx=train_dataset.class_to_idx,
        transform=EVAL_TRANSFORM,
    )

    test_dataset = PlantVillageDataset(
        PROCESSED_DIR / "test.csv",
        class_to_idx=train_dataset.class_to_idx,
        transform=EVAL_TRANSFORM,
    )

    datasets = {
        "Training": train_dataset,
        "Validation": val_dataset,
        "Test": test_dataset,
    }

    # 3. Print dataset sizes.
    print("=== DATASET SIZES ===")
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset):,} images")

    num_classes = len(train_dataset.class_to_idx)
    print("\nNumber of classes:", num_classes)

    # 4. Check that all splits use the same class mapping.
    assert train_dataset.class_to_idx == val_dataset.class_to_idx
    assert train_dataset.class_to_idx == test_dataset.class_to_idx

    # 5. Check one image from every split.
    print("\n=== INDIVIDUAL IMAGE CHECKS ===")

    for name, dataset in datasets.items():
        image, label = dataset[0]

        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert 0 <= label < num_classes

        label_name = dataset.data.iloc[0]["label"]

        print(f"\n{name}:")
        print("  Image shape:", tuple(image.shape))
        print("  Image dtype:", image.dtype)
        print("  Label index:", label)
        print("  Label name:", label_name)

    # 6. Check batch loading for all three splits.
    print("\n=== BATCH CHECKS ===")

    for name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        images, labels = next(iter(loader))

        assert images.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)
        assert labels.dtype == torch.int64

        print(f"\n{name}:")
        print("  Batch image shape:", tuple(images.shape))
        print("  Batch label shape:", tuple(labels.shape))
        print("  Batch image dtype:", images.dtype)

    # 7. Confirm all checks passed.
    print("\nSUCCESS: training, validation, and test datasets work correctly.")
    print("SUCCESS: all splits share the same class mapping.")


if __name__ == "__main__":
    main()
