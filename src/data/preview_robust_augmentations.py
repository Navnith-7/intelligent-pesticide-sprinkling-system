
import random

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision.transforms import ToPILImage

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.robust_transforms import ROBUST_TRAIN_TRANSFORM


CSV_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "augmentation_preview"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load one validation image.
df = pd.read_csv(CSV_PATH)
sample = df.iloc[0]

image_path = PROJECT_ROOT / sample["image_path"]
image = Image.open(image_path).convert("RGB")

print("Original image:", image_path.name)
print("Actual class:", sample["label"])

# Save original image.
image.resize((224, 224)).save(OUTPUT_DIR / "original.png")

# Convert normalized tensor back to a viewable image.
to_pil = ToPILImage()

def save_preview(tensor, output_path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)

    to_pil(tensor).save(output_path)


# Generate eight independently augmented versions.
for i in range(1, 9):
    augmented = ROBUST_TRAIN_TRANSFORM(image)
    output_path = OUTPUT_DIR / f"augmented_{i}.png"
    save_preview(augmented, output_path)
    print("Saved:", output_path.name)

print("\nPreview generation complete.")
print("Output folder:", OUTPUT_DIR)
