"""
End-to-end sprinkler: live_predict_v4 vision pipeline + pump safety state machine.

VISION (imported unchanged from src.inference.live_predict_v4 - nothing is
re-implemented, so behaviour cannot drift):
    best_mobilenet_v2_robust.pth, Mode A/B ('m'), LAB L-gamma + CLAHE glare
    recovery ('c'), white balance OFF by default ('w'), top-10% watermark
    exclusion, 5% minimum leaf area guardrail, 5-frame EMA smoothing.

HARDWARE (protocol copied from src/inference/automated_sprinkler.py and
src/hardware/test_relay.py - neither file is modified):
    USB serial, default COM6 @ 9600 baud, single ASCII byte per command:
    "1" = pump ON, "0" = pump OFF.  2 s wait after opening the port (the
    Arduino resets), then "0" is sent so the pump starts OFF.
    If the port cannot be opened (or --dry-run is given) the script runs in
    simulation mode and never touches serial.

SAFETY STATE MACHINE
    STANDBY   -> counts consecutive confirmations
                 (valid leaf >= 5% AND diseased_prob >= 0.70)
                 any HEALTHY / NO LEAF / low-probability frame resets it to 0
    SPRAYING  -> entered when confirm_count >= 3; pump ON for at most
                 PUMP_ON_SECONDS (1.5 s, same as automated_sprinkler.py)
    COOLDOWN  -> starts when the pump switches OFF; lasts COOLDOWN_SECONDS
                 (2.0 s); no spray can fire and no confirmations are counted
    The pump is switched OFF in a `finally:` block on every exit path.

Run from the project root:
    .venv\\Scripts\\python.exe -m src.inference.live_sprinkler_v4 --dry-run
    .venv\\Scripts\\python.exe -m src.inference.live_sprinkler_v4 --port COM6

Controls: m = Mode A/B | c = LAB correction AUTO/ON/OFF | w = white balance
          Q / ESC = quit
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.inference.live_predict_v4 as v4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Hardware - values taken from the existing automated_sprinkler.py.
DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 9600
ARDUINO_RESET_DELAY_SECONDS = 2.0
SERIAL_TIMEOUT_SECONDS = 1.0

# State machine.
CONFIRM_REQUIRED = 3
DISEASE_PROB_THRESHOLD = 0.70
PUMP_ON_SECONDS = 1.5       # same as MAX_PUMP_ON_SECONDS in automated_sprinkler.py
COOLDOWN_SECONDS = 2.0

STANDBY = "STANDBY"
SPRAYING = "SPRAYING"
COOLDOWN = "COOLDOWN"

MAIN_WINDOW = "Intelligent Sprinkler - V4 LIVE"
PREVIEW_WINDOW = v4.PREVIEW_WINDOW


# ---------------------------------------------------------------------------
# Arduino pump controller (real serial or simulation)
# ---------------------------------------------------------------------------
class PumpController:
    """Writes "1"/"0" to the Arduino, or simulates it. Never raises on I/O."""

    def __init__(self, port: str, baud: int, dry_run: bool, serial_factory=None):
        self.port = port
        self.baud = baud
        self.simulated = dry_run
        self.fallback_reason = None
        self.faulted = False
        self.is_on = False
        self._serial = None
        self._serial_factory = serial_factory

    # -- connection ---------------------------------------------------------
    def connect(self) -> None:
        if self.simulated:
            print("Arduino: SIMULATED (DRY RUN) - no serial port will be opened.")
            return

        try:
            factory = self._serial_factory
            if factory is None:
                import serial  # pyserial (already in requirements.txt)
                factory = serial.Serial

            print(f"Connecting to Arduino on {self.port} @ {self.baud} baud...")
            self._serial = factory(
                self.port,
                self.baud,
                timeout=SERIAL_TIMEOUT_SECONDS,
                write_timeout=SERIAL_TIMEOUT_SECONDS,
            )
            time.sleep(ARDUINO_RESET_DELAY_SECONDS)   # Arduino resets on open
            self._serial.reset_input_buffer()
            self._write("0")                          # start with pump OFF
            print("Arduino connected. Pump initialized to OFF.")
        except Exception as error:  # port missing / busy / pyserial missing
            self.fallback_reason = f"{self.port} unavailable: {error}"
            print(f"WARNING: {self.fallback_reason}")
            print("Falling back to SIMULATED (DRY RUN) mode - the pump will NOT run.")
            self._close_serial()
            self.simulated = True

    # -- low level ----------------------------------------------------------
    def _write(self, command: str) -> None:
        if command not in ("1", "0"):
            raise ValueError("Invalid Arduino command")
        self._serial.write(command.encode("ascii"))
        self._serial.flush()

    def _fault(self, error: Exception) -> None:
        self.faulted = True
        print(f"ARDUINO FAULT: {error} - spraying disabled.")

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    # -- pump commands ------------------------------------------------------
    def on(self) -> bool:
        """Switch the pump ON. Returns False if it could not be done."""
        if self.faulted:
            return False
        if self.simulated:
            print("[SIMULATED] PUMP ON  (command '1' not sent)")
            self.is_on = True
            return True
        try:
            self._write("1")
            self.is_on = True
            return True
        except Exception as error:
            self._fault(error)
            return False

    def off(self) -> None:
        """Best-effort OFF. Always attempted, even after a fault."""
        if self.simulated:
            if self.is_on:
                print("[SIMULATED] PUMP OFF (command '0' not sent)")
            self.is_on = False
            return
        if self._serial is None:
            self.is_on = False
            return
        try:
            self._write("0")
        except Exception as error:
            self._fault(error)
        self.is_on = False

    def close(self) -> None:
        self.off()
        self._close_serial()

    def status_line(self) -> tuple[str, tuple[int, int, int]]:
        if self.faulted:
            return "ARDUINO: FAULT - SPRAYING DISABLED", v4.RED
        if self.simulated:
            suffix = f" - {self.port} not available" if self.fallback_reason else ""
            return f"ARDUINO: SIMULATED (DRY RUN){suffix}", v4.YELLOW
        return f"ARDUINO: CONNECTED ({self.port} @ {self.baud})", v4.GREEN


# ---------------------------------------------------------------------------
# Spray state machine (pure logic - time is passed in, so it is testable)
# ---------------------------------------------------------------------------
class SprayStateMachine:
    PUMP_ON = "PUMP_ON"
    PUMP_OFF = "PUMP_OFF"

    def __init__(
        self,
        confirm_required: int = CONFIRM_REQUIRED,
        threshold: float = DISEASE_PROB_THRESHOLD,
        spray_seconds: float = PUMP_ON_SECONDS,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ):
        self.confirm_required = confirm_required
        self.threshold = threshold
        self.spray_seconds = spray_seconds
        self.cooldown_seconds = cooldown_seconds

        self.state = STANDBY
        self.confirm_count = 0
        self.trigger_class = None
        self.spray_end = 0.0
        self.cooldown_end = 0.0

    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.cooldown_end - now) if self.state == COOLDOWN else 0.0

    def reset_confirmations(self) -> None:
        self.confirm_count = 0

    def tick(self, now: float) -> list[str]:
        """Timer transitions only: SPRAYING->COOLDOWN, COOLDOWN->STANDBY."""
        actions = []
        if self.state == SPRAYING and now >= self.spray_end:
            actions.append(self.PUMP_OFF)
            self.state = COOLDOWN
            self.cooldown_end = now + self.cooldown_seconds
        if self.state == COOLDOWN and now >= self.cooldown_end:
            self.state = STANDBY
            self.confirm_count = 0
        return actions

    def observe(
        self,
        now: float,
        leaf_valid: bool,
        diseased_prob: float,
        disease_class: str | None,
        can_spray: bool = True,
    ) -> list[str]:
        """Feed one frame's result. Returns actions for the hardware."""
        actions = self.tick(now)

        if self.state != STANDBY:
            # Spraying / cooling down: nothing is counted, nothing can fire.
            self.confirm_count = 0
            return actions

        if can_spray and leaf_valid and diseased_prob >= self.threshold:
            self.confirm_count += 1
        else:
            self.confirm_count = 0          # HEALTHY, NO LEAF, low prob, or hardware fault

        if self.confirm_count >= self.confirm_required:
            actions.append(self.PUMP_ON)
            self.state = SPRAYING
            self.spray_end = now + self.spray_seconds
            self.trigger_class = disease_class
            self.confirm_count = 0
        return actions

    def abort_spray(self, now: float) -> list[str]:
        """Fail-safe (prediction error): cut the pump and enter cooldown."""
        self.confirm_count = 0
        if self.state == SPRAYING:
            self.state = COOLDOWN
            self.cooldown_end = now + self.cooldown_seconds
            return [self.PUMP_OFF]
        return []

    def cancel_spray(self) -> None:
        """PUMP_ON could not be executed (hardware fault): return to STANDBY."""
        self.state = STANDBY
        self.confirm_count = 0

    def hud_line(self, now: float) -> tuple[str, tuple[int, int, int]]:
        if self.state == SPRAYING:
            name = v4.format_class_name(self.trigger_class or "unknown")
            return f"PUMP STATE: SPRAYING! (Triggered on {name})", v4.RED
        if self.state == COOLDOWN:
            return (f"PUMP STATE: SAFETY COOLDOWN "
                    f"({self.cooldown_remaining(now):.1f} s remaining)"), v4.YELLOW
        return (f"PUMP STATE: STANDBY "
                f"(Confirmations: {self.confirm_count}/{self.confirm_required})"), v4.WHITE


# ---------------------------------------------------------------------------
# Vision: one frame through the exact live_predict_v4 pipeline
# ---------------------------------------------------------------------------
@dataclass
class VisionState:
    mode: str = v4.MODE_A
    wb_enabled: bool = v4.WB_ENABLED
    corr_mode_idx: int = 0            # 0 AUTO, 1 ON, 2 OFF
    smoothed: np.ndarray | None = None


def top_diseased_class(probs: np.ndarray, idx_to_class, healthy_idx) -> str:
    """Highest-probability NON-healthy class (used to label a spray trigger)."""
    masked = probs.copy()
    masked[healthy_idx] = -1.0
    return idx_to_class[int(np.argmax(masked))]


def analyze_frame(frame, vs: VisionState, model, device, idx_to_class, healthy_idx) -> dict:
    """Mirrors the per-frame logic of live_predict_v4.main() using its functions."""
    guide = v4.guide_box(frame.shape)
    band_px = v4.watermark_band_px(frame.shape[0])
    view = frame.copy()
    preview = v4.placeholder_preview()

    contour, area_fraction, filled, centroid = v4.detect_leaf(frame, guide)
    corr_mode = v4.CORRECTION_MODES[vs.corr_mode_idx]

    out = {
        "view": view, "preview": preview, "lines": [],
        "leaf_valid": False, "diseased_prob": 0.0, "trigger_class": None,
        "error": None, "area_fraction": area_fraction, "summary": None,
    }

    if contour is None or area_fraction < v4.MIN_LEAF_AREA_FRACTION:
        vs.smoothed = None
        v4.draw_overlays(view, guide, band_px, None, None, None)
        out["lines"] = [
            (v4.NO_LEAF_MESSAGE, v4.RED),
            (f"Leaf area: {area_fraction:.1%} (min {v4.MIN_LEAF_AREA_FRACTION:.0%})", v4.WHITE),
        ]
        return out

    roi_box = v4.square_roi_box(contour, frame.shape, band_px, pad_ratio=v4.PAD_RATIO)
    x1, y1, x2, y2 = roi_box
    source = frame if vs.mode == v4.MODE_A else v4.mode_b_composite(frame, filled)
    roi = source[y1:y2, x1:x2]
    leaf_mask_roi = filled[y1:y2, x1:x2]

    glare_ratio, mean_v = v4.measure_glare(roi, leaf_mask_roi)
    auto_trigger = glare_ratio > v4.GLARE_CORRECT_RATIO or mean_v > v4.OVEREXPOSED_MEAN_V
    correction_on = corr_mode == "ON" or (corr_mode == "AUTO" and auto_trigger)

    if correction_on:
        roi_corrected, wb_gains = v4.correct_exposure_lab(
            roi, leaf_mask_roi, wb_enabled=vs.wb_enabled)
    elif vs.wb_enabled:
        roi_corrected, wb_gains = v4.apply_white_balance(roi, leaf_mask_roi)
    else:
        roi_corrected, wb_gains = roi, None

    out["preview"] = cv2.resize(
        roi_corrected, (v4.IMAGE_SIZE, v4.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    v4.draw_overlays(view, guide, band_px, contour, roi_box, centroid)

    try:
        probs, inference_ms = v4.predict_probs(roi_corrected, model, device)
        vs.smoothed = v4.update_ema(vs.smoothed, probs)
        s = v4.summarize(vs.smoothed, idx_to_class, healthy_idx)
        (t1, p1), (t2, p2) = s["top1"], s["top2"]
        gains_txt = ("off" if wb_gains is None
                     else "B{:.2f} G{:.2f} R{:.2f}".format(*wb_gains))

        out["summary"] = s
        out["leaf_valid"] = True
        out["diseased_prob"] = s["diseased_prob"]
        out["trigger_class"] = top_diseased_class(vs.smoothed, idx_to_class, healthy_idx)
        out["lines"] = [
            (f"Top-1: {v4.format_class_name(t1)}  ({p1:.1%})", v4.GREEN),
            (f"Top-2: {v4.format_class_name(t2)}  ({p2:.1%})", v4.WHITE),
            (f"Diseased {s['diseased_prob']:.1%} | Healthy {s['healthy_prob']:.1%} "
             f"| spray threshold {DISEASE_PROB_THRESHOLD:.0%}",
             v4.RED if s["diseased_prob"] >= DISEASE_PROB_THRESHOLD else v4.GREEN),
            (f"Glare {glare_ratio:.1%} | mean V {mean_v:.0f} | "
             f"LAB L-gamma+CLAHE {'ON' if correction_on else 'OFF'} | WB {gains_txt}", v4.WHITE),
            (f"Inference {inference_ms:.1f} ms | leaf area {area_fraction:.1%} "
             f"| pad {v4.PAD_RATIO:.0%} | EMA 0.6/0.4", v4.GRAY_TXT),
        ]
    except Exception as error:  # keep the camera loop alive; fail safe upstream
        vs.smoothed = None
        out["error"] = str(error)
        out["lines"] = [("Prediction error", v4.RED), (str(error)[:90], v4.WHITE)]

    if glare_ratio > v4.GLARE_WARN_RATIO:
        out["lines"].append((v4.GLARE_WARNING, v4.RED))
    return out


# ---------------------------------------------------------------------------
# Hardware action dispatch
# ---------------------------------------------------------------------------
def apply_actions(actions, sm: SprayStateMachine, pump: PumpController, detail: str = "") -> None:
    for action in actions:
        if action == SprayStateMachine.PUMP_ON:
            print(f"\nDisease confirmed: {sm.trigger_class} {detail}")
            if pump.on():
                print(f"Pump ON for a maximum of {sm.spray_seconds:.1f} s.")
            else:
                sm.cancel_spray()
                print("Pump could not be switched ON.")
        elif action == SprayStateMachine.PUMP_OFF:
            pump.off()
            print("Pump switched OFF. Safety cooldown started.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live plant-disease sprinkler (V4 vision).")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Arduino serial port (default {DEFAULT_PORT}, from automated_sprinkler.py)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Serial baud rate (default {DEFAULT_BAUD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate the Arduino; never open a serial port")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pump = PumpController(args.port, args.baud, args.dry_run)
    camera = None
    sm = SprayStateMachine()
    vs = VisionState()

    try:
        pump.connect()

        print("\nInitializing live sprinkler (V4 vision, ROBUST MODEL)...")
        try:
            model, idx_to_class = v4.load_model(device)
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"ERROR: {error}")
            return
        healthy_idx = v4.healthy_indices(idx_to_class)

        # Warm up CUDA so the first live frame does not stall the loop
        # (the pump OFF timer is checked once per loop iteration).
        v4.predict_probs(np.zeros((v4.IMAGE_SIZE, v4.IMAGE_SIZE, 3), np.uint8), model, device)
        print(f"Model loaded and warmed up. Device: {device} | Classes: {len(idx_to_class)}")

        print("\nOpening DroidCam...")
        camera = cv2.VideoCapture(v4.CAMERA_INDEX, cv2.CAP_MSMF)
        if not camera.isOpened():
            print("ERROR: Could not open DroidCam.")
            return

        print("DroidCam connected. Center a leaf inside the yellow guide box.")
        print("Controls: 'm'=Mode A/B | 'c'=LAB correction | 'w'=white balance | 'q'/ESC=quit\n")

        windows_positioned = False

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("WARNING: Could not read a camera frame.")
                break

            # Pump timers first, before any slow vision work.
            apply_actions(sm.tick(time.monotonic()), sm, pump)

            result = analyze_frame(frame, vs, model, device, idx_to_class, healthy_idx)
            now = time.monotonic()

            if result["error"]:
                apply_actions(sm.abort_spray(now), sm, pump)   # fail safe
            else:
                detail = f"(diseased probability {result['diseased_prob']:.1%})"
                apply_actions(
                    sm.observe(now, result["leaf_valid"], result["diseased_prob"],
                               result["trigger_class"], can_spray=not pump.faulted),
                    sm, pump, detail)

            corr_mode = v4.CORRECTION_MODES[vs.corr_mode_idx]
            mode_line = (f"Mode {vs.mode} ('m') | LAB-Corr: {corr_mode} ('c') | "
                         f"WB: {'ON' if vs.wb_enabled else 'OFF'} ('w')", v4.CYAN)
            info_lines = [mode_line, sm.hud_line(now), pump.status_line()] + result["lines"]

            canvas = v4.build_main_canvas(result["view"], info_lines)
            cv2.imshow(MAIN_WINDOW, canvas)
            cv2.imshow(PREVIEW_WINDOW, result["preview"])
            if not windows_positioned:
                cv2.moveWindow(MAIN_WINDOW, 20, 20)
                cv2.moveWindow(PREVIEW_WINDOW, canvas.shape[1] + 40, 20)
                windows_positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                vs.mode = v4.MODE_B if vs.mode == v4.MODE_A else v4.MODE_A
                vs.smoothed = None
                sm.reset_confirmations()
                print(f"Switched to Mode {vs.mode}")
            if key == ord("w"):
                vs.wb_enabled = not vs.wb_enabled
                vs.smoothed = None
                sm.reset_confirmations()
                print(f"White balance: {'ON' if vs.wb_enabled else 'OFF'}")
            if key == ord("c"):
                vs.corr_mode_idx = (vs.corr_mode_idx + 1) % len(v4.CORRECTION_MODES)
                vs.smoothed = None
                sm.reset_confirmations()
                print(f"LAB correction: {v4.CORRECTION_MODES[vs.corr_mode_idx]}")

            if (cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                break

    except Exception as error:
        print(f"\nSYSTEM ERROR: {error}")

    finally:
        # Always switch the pump OFF and release every resource.
        pump.close()
        print("Safety: PUMP OFF, serial port closed.")
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("System shutdown complete.")


if __name__ == "__main__":
    main()
