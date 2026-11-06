
from pathlib import Path
import sys
import csv
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor


def create_variations(image):
    """Create controlled image-quality variations."""

    variations = {"original": image.copy()}

    variations["low_brightness"] = ImageEnhance.Brightness(
        image
    ).enhance(0.55)

    variations["high_brightness"] = ImageEnhance.Brightness(
        image
    ).enhance(1.4)

    variations["low_contrast"] = ImageEnhance.Contrast(
        image
    ).enhance(0.6)

    variations["high_contrast"] = ImageEnhance.Contrast(
        image
    ).enhance(1.5)

    variations["slight_blur"] = image.filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )

    variations["strong_blur"] = image.filter(
        ImageFilter.GaussianBlur(radius=3.0)
    )

    variations["rotated_15_degrees"] = image.rotate(
        15,
        resample=Image.Resampling.BILINEAR,
        expand=False
    )

    variations["rotated_minus_15_degrees"] = image.rotate(
        -15,
        resample=Image.Resampling.BILINEAR,
        expand=False
    )

    # Simulate lower camera resolution.
    small = image.resize((128, 128))
    variations["low_resolution"] = small.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    # Simulate JPEG compression.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=35)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        variations["jpeg_compression"] = compressed.convert("RGB")

    return variations


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            'python src/inference/camera_robustness_test.py "image_path"'
        )
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    # The parent directory is the PlantVillage class name.
    actual_class = image_path.parent.name

    print("Loading MobileNetV2...")
    predictor = PlantDiseasePredictor()

    variations = create_variations(original)

    results = []

    print("\n========== ROBUSTNESS TEST ==========")
    print("Expected class:", actual_class)

    for name, image in variations.items():

        result = predictor.predict_image(image)

        predicted = result["predicted_class"]
        confidence = result["confidence"]

        correct = predicted == actual_class

        results.append({
            "variation": name,
            "actual_class": actual_class,
            "predicted_class": predicted,
            "confidence": round(confidence, 5),
            "correct": correct,
            "inference_ms": result["inference_ms"]
        })

        status = "CORRECT" if correct else "INCORRECT"

        print(
            f"{name:28} | "
            f"{predicted:40} | "
            f"{confidence:7.2%} | {status}"
        )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "results"
        / "camera_robustness_results.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    correct_count = sum(row["correct"] for row in results)

    print("\n=====================================")
    print(f"Correct predictions: {correct_count}/{len(results)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
