
import io
import random

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomMotionBlur:
    def __call__(self, image):
        if random.random() < 0.5:
            size = random.choice([3, 5])
            kernel = np.zeros((size, size), dtype=np.float32)

            if random.random() < 0.5:
                kernel[size // 2, :] = 1.0 / size
            else:
                kernel[:, size // 2] = 1.0 / size

            array = np.array(image)
            blurred = cv2.filter2D(array, -1, kernel)
            return Image.fromarray(blurred)

        return image


class RandomDownscale:
    def __call__(self, image):
        if random.random() < 0.5:
            width, height = image.size
            scale = random.uniform(0.5, 0.9)

            small = image.resize(
                (max(32, int(width * scale)),
                 max(32, int(height * scale))),
                Image.Resampling.BILINEAR
            )

            return small.resize(
                (width, height),
                Image.Resampling.BILINEAR
            )

        return image


class RandomJPEGCompression:
    def __call__(self, image):
        if random.random() < 0.5:
            buffer = io.BytesIO()
            quality = random.randint(40, 95)

            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            with Image.open(buffer) as compressed:
                return compressed.convert("RGB")

        return image


ROBUST_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=(0.6, 1.25),
        contrast=(0.75, 1.25),
        saturation=0.15,
        hue=0.02
    ),

    transforms.RandomApply([RandomMotionBlur()], p=0.25),
    transforms.RandomApply([RandomDownscale()], p=0.25),
    transforms.RandomApply([RandomJPEGCompression()], p=0.20),

    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
