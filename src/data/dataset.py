
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "PlantVillage-Dataset" / "raw" / "color"
)

IMAGE_SIZE = 224

# ImageNet channel statistics, used consistently for both models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Augmentation is applied to training images only.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation and test images are not augmented.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class PlantVillageDataset(Dataset):
    def __init__(self, csv_path, class_to_idx=None, transform=None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.data["label"].unique())
            self.class_to_idx = {
                label: index for index, label in enumerate(classes)
            }
        else:
            self.class_to_idx = class_to_idx

        unknown_labels = set(self.data["label"]) - set(self.class_to_idx)
        if unknown_labels:
            raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = PROJECT_ROOT / row["image_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]
        return image, label
