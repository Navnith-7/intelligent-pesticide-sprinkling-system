"""
Live inference v4 - targeted fixes from live camera testing on FREC_Scab 3061.JPG.

Key upgrades over live_predict_v3.py:
  1. Fix Leaf Tip Clipping (RESTRICT_DETECTION_TO_GUIDE = False):
     - Keeps top 10% watermark exclusion band (y < int(0.10 * H)).
     - Allows full leaf contours anywhere below the watermark band so leaf tips
       extending above the guide box are never sliced off.
     - Requires only that the contour's centroid (center of mass) lies inside
       the yellow guide box.
  2. Tight Square Padding (PAD_RATIO = 0.04):
     - Reduces bounding-box padding from 12% to 4% so the leaf fills ~92% of
       the 224x224 frame, matching PlantVillage framing and excluding bright
       monitor borders.
  3. White Balance Disabled by Default (WB_ENABLED = False, toggle with 'w'):
     - Prevents Gray-World WB from neutralizing warm purple-gray PlantVillage
       backgrounds and washing out brown/olive scab lesions.
  4. LAB L-Channel Highlight Recovery + Mild Local Contrast (toggle with 'c'):
     - Converts ROI to LAB space, applies gamma=1.45 strictly to the L channel
       (preserving A/B color hues of brown/olive scab vs green leaf tissue),
       and applies mild CLAHE (clipLimit=1.5, tileGridSize=(4,4)) on L to
       recover fine lesion texture.

Run from the project root:
    .venv\\Scripts\\python.exe -m src.inference.live_predict_v4

Controls:
    m        - toggle Mode A (Natural ROI Crop) / Mode B (Feathered Gray Mask)
    c        - cycle LAB L-channel correction (AUTO -> FORCE ON -> FORCE OFF)
    w        - toggle Gray-World White Balance (OFF by default -> ON)
    Q / ESC  - quit
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EVAL_TRANSFORM, IMAGE_SIZE, PlantVillageDataset
from src.models.mobilenet import PlantDiseaseMobileNetV2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1
MAIN_WINDOW = "Intelligent Sprinkler - V4"
PREVIEW_WINDOW = "CNN Input - roi_corrected (224x224)"

ROBUST_CHECKPOINT = PROJECT_ROOT / "models" / "best_mobilenet_v2_robust.pth"
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"  # read-only mapping check

# 1. Leaf detection & unclipped contour policy
LOWER_BOUND = np.array([18, 25, 25])
UPPER_BOUND = np.array([95, 255, 255])
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MIN_LEAF_AREA_FRACTION = 0.05

WATERMARK_BAND_FRACTION = 0.10      # top 10% (y < int(0.10 * H)) excluded
RESTRICT_DETECTION_TO_GUIDE = False # allow leaf tips outside guide box; check centroid only
GUIDE_SIZE_FRACTION = 0.62
GUIDE_BOTTOM_MARGIN_FRACTION = 0.04

# 2. Tight square padding (4% on each side)
PAD_RATIO = 0.04

# Background modes ('m' to toggle)
MODE_A = "A: NATURAL ROI CROP"
MODE_B = "B: FEATHERED GRAY MASK"
MODE_B_DILATE_PX = 15
MODE_B_BLUR_KSIZE = (15, 15)
MODE_B_GRAY = 160

# 3. White balance disabled by default ('w' to toggle)
WB_ENABLED = False
WB_REFERENCE = "background"
WB_MIN_REFERENCE_FRACTION = 0.05
WB_GAIN_LIMITS = (0.8, 1.25)

# 4. LAB L-channel highlight recovery + mild CLAHE ('c' to cycle AUTO / ON / OFF)
GLARE_V_THRESHOLD = 235
GLARE_CORRECT_RATIO = 0.05
OVEREXPOSED_MEAN_V = 175
GLARE_WARN_RATIO = 0.08
GAMMA = 1.45
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (4, 4)

CORRECTION_MODES = ("AUTO", "ON", "OFF")

# Temporal smoothing + sprinkler decision
EMA_PREV_WEIGHT = 0.6
EMA_CURR_WEIGHT = 0.4
SPRAY_DECISION_THRESHOLD = 0.5

NO_LEAF_MESSAGE = "NO LEAF DETECTED - CENTER LEAF IN GUIDE BOX / MOVE CLOSER"
GLARE_WARNING = "WARNING: SCREEN GLARE BLEACHING LEAF - DIM SCREEN OR TILT PHONE"

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
GRAY_TXT = (170, 170, 170)
PANEL_BG = (25, 25, 25)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(device: torch.device):
    """Load robust checkpoint and verify its 38-class mapping."""
    if not ROBUST_CHECKPOINT.exists():
        raise FileNotFoundError(f"Robust checkpoint not found: {ROBUST_CHECKPOINT}")

    checkpoint = torch.load(ROBUST_CHECKPOINT, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(i): name for i, name in checkpoint["idx_to_class"].items()}

    if len(class_to_idx) != 38:
        raise ValueError(f"Expected 38 classes, checkpoint has {len(class_to_idx)}.")

    if TRAIN_CSV.exists():
        dataset_mapping = PlantVillageDataset(TRAIN_CSV, transform=None).class_to_idx
        if dataset_mapping != class_to_idx:
            raise ValueError("Checkpoint class mapping != src.data.dataset mapping (train.csv).")
        print("Class mapping verified against src.data.dataset (train.csv).")
    else:
        print(f"WARNING: {TRAIN_CSV} not found - using checkpoint mapping unverified.")

    model = PlantDiseaseMobileNetV2(num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_class


def healthy_indices(idx_to_class: dict[int, str]) -> np.ndarray:
    return np.array(sorted(i for i, n in idx_to_class.items() if "healthy" in n.lower()))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def watermark_band_px(frame_h: int) -> int:
    """Top 10% watermark exclusion band: y < int(0.10 * H)."""
    return int(WATERMARK_BAND_FRACTION * frame_h)


def guide_box(frame_shape) -> tuple[int, int, int, int]:
    """Square guide box horizontally centered in the lower-middle of the frame."""
    frame_h, frame_w = frame_shape[:2]
    side = int(min(frame_w, frame_h) * GUIDE_SIZE_FRACTION)
    gy2 = frame_h - int(frame_h * GUIDE_BOTTOM_MARGIN_FRACTION)
    gy1 = max(watermark_band_px(frame_h), gy2 - side)
    gx1 = (frame_w - side) // 2
    return gx1, gy1, gx1 + side, gy2


def contour_centroid(contour: np.ndarray) -> tuple[int, int]:
    """Return (cx, cy) of a contour using image moments (fallback to bounding box center)."""
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        return cx, cy
    x, y, w, h = cv2.boundingRect(contour)
    return x + w // 2, y + h // 2


def point_in_box(pt: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    px, py = pt
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def square_roi_box(
    contour: np.ndarray,
    frame_shape,
    top_limit: int,
    pad_ratio: float = PAD_RATIO,
) -> tuple[int, int, int, int]:
    """
    Bounding box padded by pad_ratio (4% by default) on all sides, expanded to
    a square, clamped to stay inside the frame and below top_limit.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)

    side = int(round(max(w, h) * (1.0 + 2.0 * pad_ratio)))
    side = max(1, min(side, frame_w, frame_h - top_limit))

    cx, cy = x + w / 2.0, y + h / 2.0
    x1 = min(max(int(round(cx - side / 2.0)), 0), frame_w - side)
    y1 = min(max(int(round(cy - side / 2.0)), top_limit), frame_h - side)
    return x1, y1, x1 + side, y1 + side


# ---------------------------------------------------------------------------
# Segmentation (unclipped below watermark band; centroid checked against guide)
# ---------------------------------------------------------------------------
def detect_leaf(frame: np.ndarray, guide: tuple[int, int, int, int]):
    """
    Detect the leaf contour anywhere below the top 10% watermark band
    (y >= int(0.10 * H)) whose centroid lies inside the guide box.

    Returns (largest_contour | None, area_fraction, filled_mask | None, centroid | None).
    """
    frame_h, frame_w = frame.shape[:2]
    band_px = watermark_band_px(frame_h)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BOUND, UPPER_BOUND)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    # Always zero out the top 10% watermark band (y < int(0.10 * H)).
    mask[:band_px, :] = 0

    if RESTRICT_DETECTION_TO_GUIDE:
        gx1, gy1, gx2, gy2 = guide
        restricted = np.zeros_like(mask)
        restricted[gy1:gy2, gx1:gx2] = mask[gy1:gy2, gx1:gx2]
        mask = restricted

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, None, None

    # Keep only contours whose centroid lies inside the guide box.
    valid_contours = []
    for c in contours:
        centroid = contour_centroid(c)
        if point_in_box(centroid, guide):
            valid_contours.append((c, centroid))

    if not valid_contours:
        return None, 0.0, None, None

    largest, largest_centroid = max(valid_contours, key=lambda item: cv2.contourArea(item[0]))
    area_fraction = cv2.contourArea(largest) / float(frame_h * frame_w)

    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [largest], 0, 255, -1)
    return largest, area_fraction, filled, largest_centroid


def mode_b_composite(frame: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """Dilate 15 px, blur 15x15, blend onto gray 160."""
    k = 2 * MODE_B_DILATE_PX + 1
    dilated = cv2.dilate(filled_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    soft = cv2.GaussianBlur(dilated, MODE_B_BLUR_KSIZE, 0)
    alpha = soft[..., None].astype(np.float32) / 255.0
    gray_bg = np.full_like(frame, MODE_B_GRAY).astype(np.float32)
    return (frame.astype(np.float32) * alpha + gray_bg * (1.0 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Glare / Exposure + LAB L-Channel Gamma & CLAHE + Optional White Balance
# ---------------------------------------------------------------------------
def measure_glare(roi_bgr: np.ndarray, leaf_mask_roi: np.ndarray) -> tuple[float, float]:
    """(glare_ratio, mean_V) over leaf pixels (falls back to whole ROI if empty)."""
    v = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    leaf = leaf_mask_roi > 127
    values = v[leaf] if leaf.any() else v.ravel()
    return float(np.mean(values > GLARE_V_THRESHOLD)), float(np.mean(values))


def apply_white_balance(
    img_bgr: np.ndarray,
    leaf_mask_roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Optional Gray-World white balance (disabled by default)."""
    img_f = img_bgr.astype(np.float32)
    if WB_REFERENCE == "background":
        reference = leaf_mask_roi <= 127
        if reference.mean() < WB_MIN_REFERENCE_FRACTION:
            return img_bgr, None
        pixels = img_f[reference]
    else:
        pixels = img_f.reshape(-1, 3)

    means = pixels.mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), *WB_GAIN_LIMITS)
    balanced = np.clip(img_f * gains, 0, 255).astype(np.uint8)
    return balanced, gains


def correct_exposure_lab(
    roi_bgr: np.ndarray,
    leaf_mask_roi: np.ndarray,
    wb_enabled: bool = WB_ENABLED,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert ROI to LAB space, apply gamma=1.45 strictly to the L channel
    (preserving A and B color hues), and apply mild CLAHE (clipLimit=1.5,
    tileGridSize=(4,4)) on L to sharpen fine scab lesions.
    Optionally applies white balance if wb_enabled is True.
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Gamma darkening strictly on L channel
    l_gamma = np.power(l_chan.astype(np.float32) / 255.0, GAMMA) * 255.0
    l_uint8 = np.clip(l_gamma, 0, 255).astype(np.uint8)

    # 2. Mild CLAHE on L channel to recover local lesion contrast
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_uint8)

    lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
    corrected_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    wb_gains = None
    if wb_enabled:
        corrected_bgr, wb_gains = apply_white_balance(corrected_bgr, leaf_mask_roi)

    return corrected_bgr, wb_gains


# ---------------------------------------------------------------------------
# Inference + EMA smoothing
# ---------------------------------------------------------------------------
def predict_probs(roi_bgr, model, device) -> tuple[np.ndarray, float]:
    """Softmax probabilities (38,) using standard EVAL_TRANSFORM."""
    pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    tensor = EVAL_TRANSFORM(pil).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, (time.perf_counter() - start) * 1000.0


def update_ema(smoothed, current: np.ndarray) -> np.ndarray:
    if smoothed is None:
        return current.copy()
    return EMA_PREV_WEIGHT * smoothed + EMA_CURR_WEIGHT * current


def summarize(probs: np.ndarray, idx_to_class, healthy_idx) -> dict:
    order = np.argsort(probs)[::-1]
    healthy_prob = float(probs[healthy_idx].sum())
    return {
        "top1": (idx_to_class[int(order[0])], float(probs[order[0]])),
        "top2": (idx_to_class[int(order[1])], float(probs[order[1]])),
        "healthy_prob": healthy_prob,
        "diseased_prob": 1.0 - healthy_prob,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def format_class_name(name: str) -> str:
    return name.replace("___", " - ").replace("_", " ")


def put_text_fit(img, text, org, color, max_width, scale=0.56, thickness=2):
    """putText, shrinking the font until the text fits max_width."""
    while scale > 0.35 and cv2.getTextSize(text, FONT, scale, thickness)[0][0] > max_width:
        scale -= 0.03
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlays(view, guide, band_px, contour, roi_box, centroid=None):
    gx1, gy1, gx2, gy2 = guide
    cv2.line(view, (0, band_px), (view.shape[1], band_px), GRAY_TXT, 1)
    cv2.putText(
        view,
        "watermark zone (top 10% ignored)",
        (8, max(14, band_px - 6)),
        FONT,
        0.4,
        GRAY_TXT,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(view, (gx1, gy1), (gx2, gy2), YELLOW, 2)
    cv2.putText(
        view,
        "CENTER LEAF HERE (tips may extend outside)",
        (gx1 + 4, gy1 + 18),
        FONT,
        0.45,
        YELLOW,
        1,
        cv2.LINE_AA,
    )
    if contour is not None:
        bx, by, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(view, (bx, by), (bx + bw, by + bh), GREEN, 2)
    if centroid is not None:
        cv2.circle(view, centroid, 4, YELLOW, -1)
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), CYAN, 1)


def build_main_canvas(view, info_lines):
    width = view.shape[1]
    line_h = 28
    strip = np.full((line_h * len(info_lines) + 12, width, 3), PANEL_BG, dtype=np.uint8)
    for i, (text, color) in enumerate(info_lines):
        put_text_fit(strip, text, (10, 22 + i * line_h), color, width - 20)
    return np.vstack([view, strip])


def placeholder_preview():
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), MODE_B_GRAY, dtype=np.uint8)
    cv2.putText(img, "NO LEAF", (60, 118), FONT, 0.8, (60, 60, 60), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing live inference v4 (ROBUST MODEL)...")

    try:
        model, idx_to_class = load_model(device)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return
    healthy_idx = healthy_indices(idx_to_class)
    print(
        f"Model loaded. Device: {device} | Classes: {len(idx_to_class)} "
        f"| Healthy classes: {len(healthy_idx)}"
    )

    print("\nOpening DroidCam...")
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not camera.isOpened():
        print("ERROR: Could not open DroidCam.")
        camera.release()
        return
    print("DroidCam connected. Center a leaf inside the yellow guide box.")
    print("Controls: 'm'=Mode A/B | 'c'=LAB Correction (AUTO/ON/OFF) | 'w'=White Balance | 'q'/ESC=Quit\n")

    mode = MODE_A
    wb_enabled = WB_ENABLED
    corr_mode_idx = 0  # 0="AUTO", 1="ON", 2="OFF"
    smoothed = None
    windows_positioned = False

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("WARNING: Could not read a camera frame.")
                break

            guide = guide_box(frame.shape)
            band_px = watermark_band_px(frame.shape[0])
            view = frame.copy()
            preview = placeholder_preview()

            contour, area_fraction, filled, centroid = detect_leaf(frame, guide)
            corr_mode = CORRECTION_MODES[corr_mode_idx]
            mode_line = (
                f"Mode {mode} ('m') | LAB-Corr: {corr_mode} ('c') | "
                f"WB: {'ON' if wb_enabled else 'OFF'} ('w')",
                CYAN,
            )

            if contour is None or area_fraction < MIN_LEAF_AREA_FRACTION:
                smoothed = None
                draw_overlays(view, guide, band_px, None, None, None)
                info_lines = [
                    mode_line,
                    (NO_LEAF_MESSAGE, RED),
                    (f"Leaf area: {area_fraction:.1%} (min {MIN_LEAF_AREA_FRACTION:.0%})", WHITE),
                ]
            else:
                roi_box = square_roi_box(contour, frame.shape, band_px, pad_ratio=PAD_RATIO)
                x1, y1, x2, y2 = roi_box
                source = frame if mode == MODE_A else mode_b_composite(frame, filled)
                roi = source[y1:y2, x1:x2]
                leaf_mask_roi = filled[y1:y2, x1:x2]

                glare_ratio, mean_v = measure_glare(roi, leaf_mask_roi)
                auto_trigger = (glare_ratio > GLARE_CORRECT_RATIO) or (mean_v > OVEREXPOSED_MEAN_V)
                correction_on = (
                    (corr_mode == "ON")
                    or (corr_mode == "AUTO" and auto_trigger)
                )

                if correction_on:
                    roi_corrected, wb_gains = correct_exposure_lab(
                        roi, leaf_mask_roi, wb_enabled=wb_enabled
                    )
                elif wb_enabled:
                    roi_corrected, wb_gains = apply_white_balance(roi, leaf_mask_roi)
                else:
                    roi_corrected, wb_gains = roi, None

                preview = cv2.resize(
                    roi_corrected, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
                )
                draw_overlays(view, guide, band_px, contour, roi_box, centroid)

                try:
                    probs, inference_ms = predict_probs(roi_corrected, model, device)
                    smoothed = update_ema(smoothed, probs)
                    s = summarize(smoothed, idx_to_class, healthy_idx)

                    (t1, p1), (t2, p2) = s["top1"], s["top2"]
                    if s["diseased_prob"] > SPRAY_DECISION_THRESHOLD:
                        spray = (f"Sprinkler: DISEASED ({s['diseased_prob']:.1%}) - SPRAY READY", RED)
                    else:
                        spray = (f"Sprinkler: HEALTHY ({s['healthy_prob']:.1%}) - NO SPRAY", GREEN)

                    gains_txt = (
                        "off"
                        if wb_gains is None
                        else "B{:.2f} G{:.2f} R{:.2f}".format(*wb_gains)
                    )
                    info_lines = [
                        mode_line,
                        (f"Top-1: {format_class_name(t1)}  ({p1:.1%})", GREEN),
                        (f"Top-2: {format_class_name(t2)}  ({p2:.1%})", WHITE),
                        spray,
                        (
                            f"Glare {glare_ratio:.1%} | mean V {mean_v:.0f} | "
                            f"LAB L-gamma+CLAHE {'ON' if correction_on else 'OFF'} | WB {gains_txt}",
                            WHITE,
                        ),
                        (
                            f"Inference {inference_ms:.1f} ms | leaf area {area_fraction:.1%} "
                            f"| pad {PAD_RATIO:.0%} | EMA 0.6/0.4",
                            GRAY_TXT,
                        ),
                    ]
                except Exception as error:
                    smoothed = None
                    info_lines = [mode_line, ("Prediction error", RED), (str(error)[:90], WHITE)]

                if glare_ratio > GLARE_WARN_RATIO:
                    info_lines.append((GLARE_WARNING, RED))

            canvas = build_main_canvas(view, info_lines)
            cv2.imshow(MAIN_WINDOW, canvas)
            cv2.imshow(PREVIEW_WINDOW, preview)
            if not windows_positioned:
                cv2.moveWindow(MAIN_WINDOW, 20, 20)
                cv2.moveWindow(PREVIEW_WINDOW, canvas.shape[1] + 40, 20)
                windows_positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = MODE_B if mode == MODE_A else MODE_A
                smoothed = None
                print(f"Switched to Mode {mode}")
            if key == ord("w"):
                wb_enabled = not wb_enabled
                smoothed = None
                print(f"White balance toggled: {'ON' if wb_enabled else 'OFF'}")
            if key == ord("c"):
                corr_mode_idx = (corr_mode_idx + 1) % len(CORRECTION_MODES)
                smoothed = None
                print(f"LAB L-channel correction mode: {CORRECTION_MODES[corr_mode_idx]}")

            if (
                cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1
            ):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Camera released. Live inference v4 stopped.")


if __name__ == "__main__":
    main()
