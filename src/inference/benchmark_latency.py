"""
Stage-3 CUDA latency benchmark (logged to CSV).

Loads models/best_mobilenet_v2_robust.pth on CUDA through live_predict_v4.load_model
and benchmarks two paths on real validation leaf images
(data/processed/validation.csv, shuffled with seed 42):

  (a) predict_probs_forward_only  - v4.predict_probs(): MobileNetV2 forward pass + softmax
                                    (+ GPU->CPU copy), bracketed by torch.cuda.synchronize().
  (b) analyze_frame_full_pipeline - live_sprinkler_v4.analyze_frame(): HSV leaf ROI +
                                    glare/LAB check (+ LAB correction when triggered) +
                                    MobileNetV2 forward pass, bracketed by
                                    torch.cuda.synchronize().

Each validation image is placed on a neutral-gray 640x480 canvas (480x480 leaf image,
centred) so the OpenCV leaf gate sees a camera-sized frame. Only images that pass the
leaf gate (area >= MIN_LEAF_AREA_FRACTION, centroid in guide box) are used, so every
timed analyze_frame() call runs the full pipeline including the CNN.

Camera capture, Streamlit rendering and serial I/O are NOT included.

Output: reports/results/stage3_latency_benchmark.csv

Run from the project root:
    .venv\\Scripts\\python.exe -m src.inference.benchmark_latency
"""

import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.inference.live_predict_v4 as v4
from src.inference.live_sprinkler_v4 import VisionState, analyze_frame

VALIDATION_CSV = PROJECT_ROOT / "data" / "processed" / "validation.csv"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "results" / "stage3_latency_benchmark.csv"

WARMUP_RUNS = 10
BENCH_RUNS = 100
SEED = 42
FRAME_WIDTH, FRAME_HEIGHT = 640, 480


def build_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Centre the leaf image (scaled to FRAME_HEIGHT square) on a neutral-gray canvas."""
    leaf = cv2.resize(image_bgr, (FRAME_HEIGHT, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)
    x0 = (FRAME_WIDTH - FRAME_HEIGHT) // 2
    canvas[:, x0:x0 + FRAME_HEIGHT] = leaf
    return canvas


def collect_samples(needed: int):
    """Return `needed` (frame, 224x224 BGR roi) pairs whose frames pass the leaf gate."""
    df = pd.read_csv(VALIDATION_CSV).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    samples = []
    scanned = 0
    for _, row in df.iterrows():
        scanned += 1
        image = cv2.imread(str(PROJECT_ROOT / row["image_path"]))
        if image is None:
            continue
        frame = build_frame(image)
        guide = v4.guide_box(frame.shape)
        contour, area_fraction, _, _ = v4.detect_leaf(frame, guide)
        if contour is None or area_fraction < v4.MIN_LEAF_AREA_FRACTION:
            continue
        roi = cv2.resize(image, (v4.IMAGE_SIZE, v4.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        samples.append((frame, roi))
        if len(samples) >= needed:
            break
    return samples, scanned


def summarize(name: str, times_ms, notes: str, gpu: str, model_name: str) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "benchmark": name,
        "device": "cuda",
        "gpu": gpu,
        "model": model_name,
        "n_warmup": WARMUP_RUNS,
        "n_runs": int(arr.size),
        "min_ms": round(float(arr.min()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(arr.max()), 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
        "notes": notes,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - this benchmark requires a CUDA GPU.")

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    model, idx_to_class = v4.load_model(device)
    healthy_idx = v4.healthy_indices(idx_to_class)
    model_name = v4.ROBUST_CHECKPOINT.name

    needed = WARMUP_RUNS + BENCH_RUNS
    samples, scanned = collect_samples(needed)
    if len(samples) < needed:
        raise SystemExit(f"Only {len(samples)} leaf-gate-passing images found (need {needed}).")
    warm, bench = samples[:WARMUP_RUNS], samples[WARMUP_RUNS:]
    print(f"GPU: {gpu} | model: {model_name} | scanned {scanned} validation images "
          f"to find {needed} that pass the leaf gate")

    # (a) model forward pass only (predict_probs synchronizes internally).
    for _, roi in warm:
        v4.predict_probs(roi, model, device)
    forward_ms = []
    for _, roi in bench:
        _, ms = v4.predict_probs(roi, model, device)
        forward_ms.append(ms)

    # (b) full analyze_frame pipeline, wall-clock bracketed by synchronize().
    for frame, _ in warm:
        analyze_frame(frame, VisionState(), model, device, idx_to_class, healthy_idx)
    full_ms = []
    ran_cnn = 0
    for frame, _ in bench:
        state = VisionState()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = analyze_frame(frame, state, model, device, idx_to_class, healthy_idx)
        torch.cuda.synchronize()
        full_ms.append((time.perf_counter() - start) * 1000.0)
        if out["summary"] is not None:
            ran_cnn += 1

    note_common = (f"{BENCH_RUNS} validation images (seed {SEED}) after {WARMUP_RUNS} warm-up runs; "
                   "camera/UI/serial excluded")
    rows = [
        summarize("predict_probs_forward_only", forward_ms,
                  f"{note_common}; forward pass + softmax + GPU->CPU copy", gpu, model_name),
        summarize("analyze_frame_full_pipeline", full_ms,
                  f"{note_common}; HSV ROI + glare/LAB check + forward pass; "
                  f"CNN ran on {ran_cnn}/{BENCH_RUNS} timed frames", gpu, model_name),
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print("\n=== STAGE-3 CUDA LATENCY BENCHMARK ===")
    print(pd.DataFrame(rows).drop(columns=["notes", "os", "torch", "cuda"]).to_string(index=False))
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
