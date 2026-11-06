
from pathlib import Path
import sys
import csv
import io
import time

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor


def make_variations(image):
    """Create controlled camera-condition simulations."""

    original = np.array(image.convert("RGB"))
    h, w = original.shape[:2]

    variants = {"original": original.copy()}

    # 1. Motion blur.
    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, :] = 1.0 / 9
    variants["motion_blur"] = cv2.filter2D(
        original, -1, kernel
    )

    # 2. Uneven shadow.
    shadow = original.copy()
    overlay = np.zeros_like(shadow)
    cv2.rectangle(
        overlay,
        (0, 0),
        (w // 2, h),
        (35, 35, 35),
        -1
    )
    variants["partial_shadow"] = cv2.addWeighted(
        shadow, 1.0, overlay, 0.55, 0
    )

    # 3. Warm colour temperature.
    warm = original.astype(np.float32)
    warm[:, :, 0] *= 1.12
    warm[:, :, 1] *= 1.03
    warm[:, :, 2] *= 0.82
    variants["warm_light"] = np.clip(
        warm, 0, 255
    ).astype(np.uint8)

    # 4. Cool colour temperature.
    cool = original.astype(np.float32)
    cool[:, :, 0] *= 0.82
    cool[:, :, 1] *= 1.02
    cool[:, :, 2] *= 1.15
    variants["cool_light"] = np.clip(
        cool, 0, 255
    ).astype(np.uint8)

    # 5. Low light with sensor noise.
    dark = original.astype(np.float32) * 0.40
    noise = np.random.default_rng(42).normal(
        0, 10, original.shape
    )
    variants["low_light_noise"] = np.clip(
        dark + noise, 0, 255
    ).astype(np.uint8)

    # 6. Stronger perspective distortion.
    source_points = np.float32([
        [0, 0], [w - 1, 0],
        [0, h - 1], [w - 1, h - 1]
    ])

    destination_points = np.float32([
        [int(w * 0.12), int(h * 0.05)],
        [int(w * 0.88), 0],
        [0, int(h * 0.90)],
        [w - 1, h - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        source_points, destination_points
    )

    variants["perspective"] = cv2.warpPerspective(
        original, matrix, (w, h)
    )

    # 7. Partial occlusion.
    occluded = original.copy()
    cv2.rectangle(
        occluded,
        (int(w * 0.65), int(h * 0.35)),
        (w - 1, int(h * 0.65)),
        (90, 90, 90),
        -1
    )
    variants["partial_occlusion"] = occluded

    # 8. Mild glare.
    glare = original.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = int(w * 0.75), int(h * 0.25)

    distance = np.sqrt(
        ((xx - cx) / max(w, 1)) ** 2
        + ((yy - cy) / max(h, 1)) ** 2
    )

    glow = np.clip(1.0 - distance * 3.0, 0, 1)
    glare += glow[:, :, None] * 75

    variants["glare"] = np.clip(
        glare, 0, 255
    ).astype(np.uint8)

    # 9. Reduced camera resolution.
    small = cv2.resize(
        original, (max(32, w // 4), max(32, h // 4))
    )
    variants["low_resolution"] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_LINEAR
    )

    # 10. JPEG compression.
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 25]
    )

    if success:
        decoded = cv2.imdecode(
            encoded, cv2.IMREAD_COLOR
        )
        variants["jpeg_compression"] = cv2.cvtColor(
            decoded, cv2.COLOR_BGR2RGB
        )

    return variants


def image_quality(image):
    """Measure basic image clarity and exposure."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2)
    }


def predict_top3(predictor, image):
    """Return top 3 model predictions."""

    pil_image = Image.fromarray(image).convert("RGB")
    tensor = predictor.transform(pil_image).unsqueeze(0)
    tensor = tensor.to(predictor.device)

    start = time.perf_counter()

    with torch.inference_mode():
        if predictor.device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = predictor.model(tensor)
        else:
            logits = predictor.model(tensor)

        probabilities = torch.softmax(
            logits.float(), dim=1
        )[0]

        values, indices = torch.topk(
            probabilities, k=3
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    predictions = []

    for value, index in zip(values, indices):
        class_name = predictor.idx_to_class[
            int(index.item())
        ]

        predictions.append(
            (class_name, float(value.item()))
        )

    return predictions, round(elapsed_ms, 2)


def main():

    if len(sys.argv) < 2:
        print(
            'Usage: python real_world_robustness_test.py "image_path"'
        )
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    actual_class = image_path.parent.name

    print("Loading MobileNetV2...")
    predictor = PlantDiseasePredictor()

    variants = make_variations(original)

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "robustness_previews"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        PROJECT_ROOT
        / "reports"
        / "results"
        / "real_world_robustness.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nEXPECTED CLASS:", actual_class)
    print("\n========== CAMERA ROBUSTNESS ==========\n")

    for name, image in variants.items():

        predictions, inference_ms = predict_top3(
            predictor, image
        )

        top1_class, top1_confidence = predictions[0]
        quality = image_quality(image)

        correct = top1_class == actual_class

        status = "CORRECT" if correct else "INCORRECT"

        print(f"\n{name}")
        print(f"  Top 1: {top1_class} ({top1_confidence:.2%})")
        print(
            f"  Top 2: {predictions[1][0]} "
            f"({predictions[1][1]:.2%})"
        )
        print(
            f"  Top 3: {predictions[2][0]} "
            f"({predictions[2][1]:.2%})"
        )
        print(
            f"  Sharpness: {quality['sharpness']}"
        )
        print(
            f"  Brightness: {quality['brightness']}"
        )
        print(
            f"  Contrast: {quality['contrast']}"
        )
        print(f"  Result: {status}")

        results.append({
            "variation": name,
            "actual_class": actual_class,
            "predicted_class": top1_class,
            "top1_confidence": round(top1_confidence, 5),
            "top2_class": predictions[1][0],
            "top2_confidence": round(predictions[1][1], 5),
            "top3_class": predictions[2][0],
            "top3_confidence": round(predictions[2][1], 5),
            "correct": correct,
            **quality,
            "inference_ms": inference_ms
        })

        preview_path = output_dir / f"{name}.jpg"
        cv2.imwrite(
            str(preview_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    correct_count = sum(
        row["correct"] for row in results
    )

    print("\n========== SUMMARY ==========")
    print(f"Correct: {correct_count}/{len(results)}")
    print(f"CSV: {csv_path}")
    print(f"Image previews: {output_dir}")


if __name__ == "__main__":
    main()
