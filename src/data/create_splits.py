
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "plantvillage_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def main():
    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {"image_path", "label", "sha256"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {sorted(missing_columns)}")

    if df[list(required_columns)].isnull().any().any():
        raise ValueError("Manifest contains missing paths, labels, or hashes.")

    # Every exact image hash must have exactly one label.
    labels_per_hash = df.groupby("sha256")["label"].nunique()
    conflicting_hashes = labels_per_hash[labels_per_hash > 1]
    if not conflicting_hashes.empty:
        raise ValueError(
            f"{len(conflicting_hashes)} exact image hashes have conflicting labels."
        )

    # Each hash is an indivisible group. Shuffle groups deterministically
    # within each class, then assign whole groups to each split.
    rng = np.random.default_rng(RANDOM_SEED)
    split_frames = {"train": [], "validation": [], "test": []}

    for label, class_df in df.groupby("label", sort=True):
        groups = class_df.groupby("sha256", sort=False)
        group_ids = list(groups.groups.keys())
        rng.shuffle(group_ids)

        total_images = len(class_df)
        target_train = round(total_images * SPLIT_RATIOS[0])
        target_val = round(total_images * SPLIT_RATIOS[1])

        counts = {"train": 0, "validation": 0, "test": 0}

        for hash_id in group_ids:
            group_df = groups.get_group(hash_id)
            group_size = len(group_df)

            # Prefer the split furthest below its target proportion.
            targets = {
                "train": target_train,
                "validation": target_val,
                "test": total_images - target_train - target_val,
            }
            available = [
                name for name in split_frames
                if counts[name] < targets[name]
            ]

            if not available:
                available = list(split_frames)

            chosen = max(
                available,
                key=lambda name: (
                    targets[name] - counts[name],
                    -counts[name],
                ),
            )

            split_frames[chosen].append(group_df)
            counts[chosen] += group_size

    splits = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in split_frames.items()
    }

    # Validate class coverage and exact-hash isolation.
    all_classes = set(df["label"].unique())

    for name, data in splits.items():
        missing = all_classes - set(data["label"].unique())
        if missing:
            raise RuntimeError(
                f"{name} is missing classes: {sorted(missing)}"
            )

    for name_a, data_a in splits.items():
        for name_b, data_b in splits.items():
            if name_a < name_b:
                overlap = set(data_a["sha256"]) & set(data_b["sha256"])
                if overlap:
                    raise RuntimeError(
                        f"Duplicate leakage between {name_a} and {name_b}"
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        data[["image_path", "label", "sha256"]].to_csv(path, index=False)
        print(
            f"{name.capitalize():10} {len(data):,} images "
            f"({len(data) / len(df) * 100:.2f}%) -> {path}"
        )

    print("\nClass counts per split:")
    counts = pd.DataFrame({
        name: data["label"].value_counts()
        for name, data in splits.items()
    }).fillna(0).astype(int)
    print(counts.to_string())

    print("\nSUCCESS: every class appears in every split.")
    print("SUCCESS: no exact image hashes cross split boundaries.")
    print("NOTE: split ratios are approximate because duplicate groups stay intact.")


if __name__ == "__main__":
    main()
