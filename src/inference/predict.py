
from pathlib import Path
import sys
import time

import torch
from PIL import Image

# Ensure the project root is available for imports when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EVAL_TRANSFORM
from src.models.mobilenet import PlantDiseaseMobileNetV2

MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenet_v2_finetuned.pth"


class PlantDiseasePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {
            int(index): label
            for index, label in checkpoint["idx_to_class"].items()
        }

        self.model = PlantDiseaseMobileNetV2(
            num_classes=len(self.class_to_idx),
            pretrained=False,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = EVAL_TRANSFORM

        print("Model loaded successfully.")
        print(f"Device: {self.device}")
        print(f"Classes: {len(self.class_to_idx)}")

    def predict_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("predict_image() expects a PIL Image.")

        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(self.device.type == "cuda"),
            ):
                logits = self.model(image_tensor)

            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predicted_idx = probabilities.max(dim=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - start_time) * 1000
        predicted_idx = predicted_idx.item()

        return {
            "predicted_class": self.idx_to_class[predicted_idx],
            "confidence": float(confidence.item()),
            "inference_ms": round(inference_ms, 2),
            "model": "MobileNetV2-finetuned",
        }

    def predict(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            return self.predict_image(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict a plant disease from an image."
    )
    parser.add_argument("image", nargs="?", help="Path to the leaf image")
    args = parser.parse_args()

    predictor = PlantDiseasePredictor()

    if args.image is None:
        print("\nUsage:")
        print(
            r'.\.venv\Scripts\python.exe src\inference\predict.py "path\to\leaf.jpg"'
        )
    else:
        result = predictor.predict(args.image)

        print("\n========== PREDICTION RESULT ==========")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.2%}")
        print(f"Inference time:  {result['inference_ms']:.2f} ms")
        print(f"Model:           {result['model']}")
        print("=======================================")
