"""
Precision Foliar Sprayer â€” Real-Time Vision & Hardware Control Console
Domain-Robust MobileNetV2 + OpenCV V4 ROI/LAB Optics + Arduino Uno Relay Controller
"""

import html
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

try:
    import serial
except ImportError:
    serial = None

from src.inference import live_predict_v4 as v4
from src.inference.live_sprinkler_v4 import VisionState, analyze_frame
from src.inference.pesticide_recommendation import (
    PESTICIDE_DB,
    estimate_leaf_severity_and_duration,
    get_pesticide_recommendation,
)
from src.utils.plot_benchmarks import FIGURES_DIR, generate_all_charts

st.set_page_config(
    page_title="Foliar Sprayer Control Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --bg-base: #0b0e14;
        --bg-panel: #12161f;
        --bg-panel-alt: #161b26;
        --border-subtle: #232a38;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-green: #2ea043;
        --accent-amber: #d29922;
        --accent-red: #f85149;
        --accent-blue: #58a6ff;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .stApp {
        background-color: var(--bg-base);
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    /* Hide default Streamlit deploy/menu chrome for cleaner instrumentation look */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    section[data-testid="stSidebar"] {
        background-color: #0e121a;
        border-right: 1px solid var(--border-subtle);
    }

    /* Top Header Bar */
    .console-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    .console-title {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 0;
    }
    .console-meta {
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--text-secondary);
    }

    /* Unified 5-Tile Telemetry Strip */
    .telemetry-grid {
        display: grid;
        grid-template-columns: 1.35fr 0.9fr 1.15fr 0.95fr 0.95fr;
        gap: 10px;
        margin: 0.65rem 0 1rem 0;
    }
    .t-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 82px;
    }
    .t-card.state-standby { border-left: 3px solid var(--accent-green); }
    .t-card.state-spraying { border-left: 3px solid var(--accent-red); background: #1a1216; }
    .t-card.state-cooldown { border-left: 3px solid var(--accent-amber); background: #191610; }
    .t-card.state-locked { border-left: 3px solid var(--accent-blue); background: #111621; }

    .t-label {
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        font-weight: 600;
    }
    .t-value {
        font-family: var(--mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 3px 0 2px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .t-sub {
        font-family: var(--mono);
        font-variant-numeric: tabular-nums;
        font-size: 0.74rem;
        color: var(--text-secondary);
    }
    .progress-track {
        width: 100%;
        height: 4px;
        background: #232a38;
        border-radius: 2px;
        margin-top: 6px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 2px;
    }

    /* Prescription / Diagnosis Panel */
    .rx-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
    }
    .rx-panel.diseased { border-top: 2px solid var(--accent-red); }
    .rx-panel.healthy { border-top: 2px solid var(--accent-green); }
    .rx-panel.idle { border-top: 2px solid #3b4454; }

    .rx-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .rx-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .pill {
        font-family: var(--mono);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pill-red { background: rgba(248, 81, 73, 0.14); color: #ff7b72; border: 1px solid rgba(248, 81, 73, 0.35); }
    .pill-green { background: rgba(46, 160, 67, 0.14); color: #56d364; border: 1px solid rgba(46, 160, 67, 0.35); }
    .pill-blue { background: rgba(88, 166, 255, 0.14); color: #79c0ff; border: 1px solid rgba(88, 166, 255, 0.35); }
    .pill-muted { background: rgba(139, 148, 158, 0.12); color: #8b949e; border: 1px solid var(--border-subtle); }

    .kv-grid {
        display: grid;
        grid-template-columns: 135px 1fr;
        row-gap: 7px;
        column-gap: 10px;
        font-size: 0.84rem;
    }
    .kv-k {
        color: var(--text-secondary);
        font-weight: 500;
    }
    .kv-v {
        color: var(--text-primary);
        font-weight: 500;
    }
    .kv-v.mono {
        font-family: var(--mono);
        font-variant-numeric: tabular-nums;
    }

    .section-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-secondary);
        font-weight: 600;
        margin: 0.2rem 0 0.5rem 0;
    }

    .policy-bar {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 8px 14px;
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        height: 38px;
    }
    .policy-bar b { color: var(--text-primary); font-weight: 600; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.86rem;
        font-weight: 500;
        padding: 8px 4px;
    }
</style>
"""


class LatestFrameCamera:
    """Background daemon reader that drops stale buffered frames for low-latency streaming."""

    def __init__(self, cam_index: int):
        self.cap = cv2.VideoCapture(int(cam_index), cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(int(cam_index))
        self.opened = self.cap.isOpened()
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False

        if self.opened:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, first = self.cap.read()
            if ok:
                self.frame = first
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        if self.opened and hasattr(self, "thread"):
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()


class SafeSerialPump:
    """Fail-safe Arduino Uno serial controller ('1' = ON, '0' = OFF) with dry-run fallback."""

    def __init__(self, port: str, baud: int = 9600, dry_run: bool = True):
        self.port = port
        self.baud = int(baud)
        self.dry_run = bool(dry_run)
        self.ser = None
        self.is_on = False
        self.fault = False
        self.status_text = "DRY-RUN"

        if not self.dry_run:
            if serial is None:
                self.dry_run = True
                self.status_text = "DRY-RUN (no pyserial)"
            else:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=1.0, write_timeout=1.0)
                    time.sleep(2.0)  # Arduino Uno bootloader reset delay
                    self.ser.reset_input_buffer()
                    self.ser.write(b"0")
                    self.ser.flush()
                    self.status_text = f"{self.port} @ {self.baud}"
                except Exception as exc:
                    self.dry_run = True
                    self.ser = None
                    self.status_text = f"DRY-RUN ({self.port} offline)"
                    print(f"[Pump Fallback] {exc}")

    def turn_on(self, reason: str = "") -> None:
        self.is_on = True
        if self.dry_run or self.ser is None:
            print(f"[SIMULATED] PUMP ON ('1') -> {reason}")
            return
        try:
            self.ser.write(b"1")
            self.ser.flush()
            print(f"[ARDUINO] PUMP ON ('1') -> {reason}")
        except Exception as exc:
            self.fault = True
            self.status_text = "SERIAL FAULT"
            print(f"[ARDUINO FAULT] {exc}")

    def turn_off(self, reason: str = "") -> None:
        self.is_on = False
        if self.dry_run or self.ser is None:
            print(f"[SIMULATED] PUMP OFF ('0') -> {reason}")
            return
        try:
            self.ser.write(b"0")
            self.ser.flush()
            print(f"[ARDUINO] PUMP OFF ('0') -> {reason}")
        except Exception as exc:
            self.fault = True
            self.status_text = "SERIAL FAULT"
            print(f"[ARDUINO FAULT] {exc}")

    def close(self) -> None:
        self.turn_off("shutdown")
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


class LeafBoundedSprayStateMachine:
    """
    Fail-Safe State Machine with:
      - Consecutive disease frame confirmations (>= threshold)
      - Variable-rate adaptive pulse duration (1.0s Mild / 1.5s Moderate / 2.0s Severe)
      - Hard per-leaf max spray cap (e.g., 2/2 LOCKED until leaf removed for reset_clear_sec)
      - Non-blocking real-time cooldown timer
    """

    def __init__(
        self,
        confirm_required: int = 3,
        disease_prob_threshold: float = 0.70,
        default_pump_on_seconds: float = 1.5,
        cooldown_seconds: float = 2.0,
        max_sprays_per_leaf: int = 2,
        reset_clear_sec: float = 1.0,
    ):
        self.confirm_required = confirm_required
        self.disease_prob_threshold = disease_prob_threshold
        self.default_pump_on_seconds = default_pump_on_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_sprays_per_leaf = max_sprays_per_leaf
        self.reset_clear_sec = reset_clear_sec

        self.state = "STANDBY"  # STANDBY, SPRAYING, COOLDOWN, MAX_REACHED
        self.confirm_count = 0
        self.clear_since = None
        self.leaf_spray_count = 0
        self.total_session_sprays = 0
        self.active_pulse_sec = default_pump_on_seconds
        self.state_started = 0.0
        self.state_until = 0.0
        self.trigger_label = ""
        self.last_pesticide = "None"

        self.healthy_frames_skipped = 0
        self.diseased_frames_seen = 0
        self.total_ml_dispensed = 0.0

    def reset_for_new_leaf(self) -> None:
        self.state = "STANDBY"
        self.confirm_count = 0
        self.clear_since = None
        self.leaf_spray_count = 0
        self.trigger_label = ""

    def tick_timers(self, now: float) -> str:
        """Fast timer check used during SPRAYING and COOLDOWN without running CNN inference."""
        if self.state == "SPRAYING":
            if now >= self.state_until:
                self.state = "COOLDOWN"
                self.state_started = now
                self.state_until = now + self.cooldown_seconds
                self.confirm_count = 0
                return "TURN_OFF"
            return "NONE"

        if self.state == "COOLDOWN":
            if now >= self.state_until:
                if self.leaf_spray_count >= self.max_sprays_per_leaf:
                    self.state = "MAX_REACHED"
                else:
                    self.state = "STANDBY"
                self.confirm_count = 0
                self.clear_since = None
            return "NONE"

        return "NONE"

    def step(
        self,
        now: float,
        leaf_valid: bool,
        diseased_prob: float,
        diseased_label: str,
        pesticide_name: str,
        pulse_sec: float,
        est_volume_ml: float,
    ) -> str:
        """Advance state machine one inference step. Returns 'NONE', 'TURN_ON', or 'TURN_OFF'."""
        if self.state in ("SPRAYING", "COOLDOWN"):
            return self.tick_timers(now)

        if leaf_valid:
            if diseased_prob >= self.disease_prob_threshold:
                self.diseased_frames_seen += 1
            else:
                self.healthy_frames_skipped += 1

        is_diseased_frame = leaf_valid and (diseased_prob >= self.disease_prob_threshold)
        if not is_diseased_frame:
            self.confirm_count = 0
            if self.leaf_spray_count > 0 or self.state == "MAX_REACHED":
                if self.clear_since is None:
                    self.clear_since = now
                elif (now - self.clear_since) >= self.reset_clear_sec:
                    self.reset_for_new_leaf()
            return "NONE"

        self.clear_since = None

        if self.state == "MAX_REACHED" or self.leaf_spray_count >= self.max_sprays_per_leaf:
            self.state = "MAX_REACHED"
            self.confirm_count = 0
            return "NONE"

        self.confirm_count += 1
        if self.confirm_count >= self.confirm_required:
            self.state = "SPRAYING"
            self.active_pulse_sec = pulse_sec
            self.state_started = now
            self.state_until = now + pulse_sec
            self.confirm_count = 0
            self.leaf_spray_count += 1
            self.total_session_sprays += 1
            self.total_ml_dispensed += est_volume_ml
            self.trigger_label = diseased_label
            self.last_pesticide = pesticide_name
            return "TURN_ON"

        return "NONE"

    @property
    def chemical_saved_pct(self) -> float:
        total = self.healthy_frames_skipped + self.diseased_frames_seen
        if total == 0:
            return 0.0
        return (self.healthy_frames_skipped / total) * 100.0


def encode_jpeg(bgr_img: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def render_telemetry_html(
    sm: LeafBoundedSprayStateMachine,
    now: float,
    leaf_valid: bool,
    is_diseased: bool,
    area_frac: float,
    rec: dict,
    diseased_prob: float,
    healthy_prob: float,
    frame_ms: float,
    pulse_to_use: float,
    sev: dict,
) -> str:
    if sm.state == "SPRAYING":
        rem = max(0.0, sm.state_until - now)
        total = max(sm.active_pulse_sec, 0.1)
        pct = min(100.0, max(0.0, ((total - rem) / total) * 100.0))
        state_cls = "state-spraying"
        state_val = f"SPRAYING  {rem:.2f}s"
        state_sub = f"Pulse {sm.active_pulse_sec:.1f}s Â· Valve OPEN"
        bar_html = f'<div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%;background:#f85149;"></div></div>'
    elif sm.state == "COOLDOWN":
        rem = max(0.0, sm.state_until - now)
        total = max(sm.cooldown_seconds, 0.1)
        pct = min(100.0, max(0.0, ((total - rem) / total) * 100.0))
        state_cls = "state-cooldown"
        state_val = f"COOLDOWN  {rem:.2f}s"
        state_sub = f"Lockout {sm.cooldown_seconds:.1f}s Â· Valve CLOSED"
        bar_html = f'<div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%;background:#d29922;"></div></div>'
    elif sm.state == "MAX_REACHED":
        state_cls = "state-locked"
        state_val = f"LOCKED ({sm.leaf_spray_count}/{sm.max_sprays_per_leaf})"
        state_sub = "Remove leaf (1.0s) to reset"
        bar_html = '<div class="progress-track"><div class="progress-fill" style="width:100%;background:#58a6ff;"></div></div>'
    else:
        state_cls = "state-standby"
        state_val = "STANDBY"
        state_sub = f"Confirm gate: {sm.confirm_count}/{sm.confirm_required}"
        pct = (sm.confirm_count / max(sm.confirm_required, 1)) * 100.0
        bar_html = f'<div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%;background:#2ea043;"></div></div>'

    if not leaf_valid:
        det_val = "No Leaf"
        det_sub = f"ROI area {area_frac * 100:.1f}% (min 5%)"
        dur_val = "0.0 s"
        dur_sub = "Idle"
    elif is_diseased:
        det_val = html.escape(rec["disease"])
        det_sub = f"{diseased_prob * 100:.1f}% Â· {frame_ms:.1f} ms"
        dur_val = f"{pulse_to_use:.1f} s"
        dur_sub = f"Severity {sev['severity_pct']}% ({sev['severity_grade'].split()[0]})"
    else:
        det_val = f"{html.escape(rec['crop'])} (Healthy)"
        det_sub = f"{healthy_prob * 100:.1f}% Â· {frame_ms:.1f} ms"
        dur_val = "0.0 s"
        dur_sub = "Healthy â€” skipped"

    return f"""
    <div class="telemetry-grid">
        <div class="t-card {state_cls}">
            <div class="t-label">Actuator State</div>
            <div class="t-value">{state_val}</div>
            <div class="t-sub">{state_sub}</div>
            {bar_html}
        </div>
        <div class="t-card">
            <div class="t-label">Leaf Spray Cap</div>
            <div class="t-value">{sm.leaf_spray_count} / {sm.max_sprays_per_leaf}</div>
            <div class="t-sub">Session sprays: {sm.total_session_sprays}</div>
        </div>
        <div class="t-card">
            <div class="t-label">Classification</div>
            <div class="t-value">{det_val}</div>
            <div class="t-sub">{det_sub}</div>
        </div>
        <div class="t-card">
            <div class="t-label">Target Pulse</div>
            <div class="t-value">{dur_val}</div>
            <div class="t-sub">{dur_sub}</div>
        </div>
        <div class="t-card">
            <div class="t-label">Chemical Saved</div>
            <div class="t-value">{sm.chemical_saved_pct:.1f}%</div>
            <div class="t-sub">Dispensed: {sm.total_ml_dispensed:.1f} mL</div>
        </div>
    </div>
    """


def render_prescription_html(
    sm: LeafBoundedSprayStateMachine,
    leaf_valid: bool,
    is_diseased: bool,
    rec: dict,
    diseased_prob: float,
    healthy_prob: float,
    pulse_to_use: float,
    vol_to_use: float,
    sev: dict,
    pump_status: str,
) -> str:
    if not leaf_valid:
        return f"""
        <div class="rx-panel idle">
            <div class="rx-header">
                <div class="rx-title">Waiting for Leaf Target</div>
                <span class="pill pill-muted">No Target</span>
            </div>
            <div class="kv-grid">
                <div class="kv-k">Positioning</div>
                <div class="kv-v">Center leaf inside the 62% guide box (area &ge; 5.0%).</div>
                <div class="kv-k">Hardware Link</div>
                <div class="kv-v mono">{html.escape(pump_status)}</div>
            </div>
        </div>
        """

    if is_diseased:
        if sm.state == "MAX_REACHED":
            pill = f'<span class="pill pill-blue">Cap Locked {sm.leaf_spray_count}/{sm.max_sprays_per_leaf}</span>'
            cap_text = f"Limit reached ({sm.leaf_spray_count}/{sm.max_sprays_per_leaf}) â€” remove leaf for 1.0s to reset"
        elif sm.state == "SPRAYING":
            pill = '<span class="pill pill-red">Spraying Active</span>'
            cap_text = f"Executing pulse {sm.leaf_spray_count}/{sm.max_sprays_per_leaf}"
        elif sm.state == "COOLDOWN":
            pill = '<span class="pill pill-muted">Cooldown Lockout</span>'
            cap_text = f"{sm.leaf_spray_count}/{sm.max_sprays_per_leaf} pulses applied"
        else:
            pill = f'<span class="pill pill-red">Diseased Â· {diseased_prob * 100:.1f}%</span>'
            cap_text = f"Armed ({sm.leaf_spray_count}/{sm.max_sprays_per_leaf} pulses used)"

        return f"""
        <div class="rx-panel diseased">
            <div class="rx-header">
                <div class="rx-title">{html.escape(rec['crop'])} â€” {html.escape(rec['disease'])}</div>
                {pill}
            </div>
            <div class="kv-grid">
                <div class="kv-k">Pathogen</div>
                <div class="kv-v">{html.escape(rec['pathogen'])}</div>
                <div class="kv-k">Treatment ({html.escape(rec['group'])})</div>
                <div class="kv-v"><strong>{html.escape(rec['pesticide'])}</strong></div>
                <div class="kv-k">Active Ingredient</div>
                <div class="kv-v">{html.escape(rec['active_ingredient'])}</div>
                <div class="kv-k">Mixing Dosage</div>
                <div class="kv-v mono">{html.escape(rec['dosage'])}</div>
                <div class="kv-k">Pulse / Volume</div>
                <div class="kv-v mono">{pulse_to_use:.1f}s (~{vol_to_use:.1f} mL) Â· {sev['severity_pct']}% {html.escape(sev['severity_grade'])}</div>
                <div class="kv-k">Leaf Cap Guard</div>
                <div class="kv-v">{cap_text}</div>
                <div class="kv-k">Hardware Link</div>
                <div class="kv-v mono">{html.escape(pump_status)}</div>
            </div>
        </div>
        """

    return f"""
    <div class="rx-panel healthy">
        <div class="rx-header">
            <div class="rx-title">{html.escape(rec['crop'])} â€” Healthy Foliage</div>
            <span class="pill pill-green">Healthy Â· {healthy_prob * 100:.1f}%</span>
        </div>
        <div class="kv-grid">
            <div class="kv-k">Treatment</div>
            <div class="kv-v">{html.escape(rec['pesticide'])}</div>
            <div class="kv-k">Agronomy Note</div>
            <div class="kv-v">{html.escape(rec['notes'])}</div>
            <div class="kv-k">Hardware Link</div>
            <div class="kv-v mono">{html.escape(pump_status)} (Relay OFF)</div>
        </div>
    </div>
    """


@st.cache_resource
def load_cached_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = v4.load_model(device)
    healthy_idx = v4.healthy_indices(idx_to_class)
    return model, idx_to_class, healthy_idx, device


def ensure_benchmark_charts() -> None:
    expected = [
        FIGURES_DIR / "model_progression_4stage.png",
        FIGURES_DIR / "domain_shift_190_comparison.png",
        FIGURES_DIR / "stage3_training_curves.png",
    ]
    if not all(p.exists() for p in expected):
        generate_all_charts()


def get_raw_class_from_probs(probs: np.ndarray, idx_to_class: dict, healthy_idx: set, is_diseased: bool) -> str:
    if probs is None:
        return "Apple___healthy"
    order = np.argsort(probs)[::-1]
    if is_diseased:
        for i in order:
            if int(i) not in healthy_idx:
                return idx_to_class[int(i)]
    return idx_to_class[int(order[0])]


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    ensure_benchmark_charts()
    model, idx_to_class, healthy_idx, device = load_cached_model()

    if "spray_log" not in st.session_state:
        st.session_state.spray_log = []
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    st.markdown(
        f"""
        <div class="console-header">
            <div class="console-title">Intelligent Pesticide Sprinkling System</div>
            <div class="console-meta">MobileNetV2-Robust (99.37% val) &nbsp;|&nbsp; OpenCV V4 ROI + LAB &nbsp;|&nbsp; {str(device).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    corr_modes = list(v4.CORRECTION_MODES)
    mode_b_val = getattr(v4, "MODE_B", "MODE_B")

    with st.sidebar:
        st.markdown('<div class="section-label">Optical Pipeline</div>', unsafe_allow_html=True)
        roi_mode_label = st.selectbox(
            "ROI Background Mode",
            ["Mode A: Natural Crop", "Mode B: Feathered Gray (160)"],
            index=0,
        )
        mode_b_enabled = "Mode B" in roi_mode_label
        glare_mode = st.selectbox("LAB L-Channel Glare Recovery", corr_modes, index=0)
        wb_enabled = st.checkbox("Gray-World White Balance", value=False)

        st.divider()
        st.markdown('<div class="section-label">Actuation & Safety Gate</div>', unsafe_allow_html=True)
        max_sprays_per_leaf = st.number_input("Max Sprays Per Leaf", min_value=1, max_value=5, value=2)
        adaptive_duration = st.checkbox(
            "Adaptive Severity Duration",
            value=True,
            help="Mild <12%: 1.0s | Moderate 12-30%: 1.5s | Severe >30%: 2.0s",
        )
        fixed_pump_sec = st.number_input("Fixed Pulse (s)", min_value=0.5, max_value=4.0, value=1.5, step=0.5)
        confirm_frames = st.slider("Confirmation Frames", 1, 5, 3)
        disease_thresh = st.slider("Disease Threshold (%)", 50, 95, 70) / 100.0
        cooldown_sec = st.number_input("Cooldown Lockout (s)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

        st.divider()
        st.markdown('<div class="section-label">Camera & Arduino Serial</div>', unsafe_allow_html=True)
        cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=1)
        serial_port = st.text_input("Serial Port", value="COM6")
        baud_rate = st.number_input("Baud Rate", value=9600, step=1200)
        dry_run_flag = st.checkbox("Dry-Run Simulation", value=True)

        if st.button("Clear Session Log", width="stretch"):
            st.session_state.spray_log = []
            st.rerun()

    tab_live, tab_upload, tab_bench, tab_formulary = st.tabs(
        ["Live Control", "Image Diagnostics", "Model Benchmarks", "Pesticide Formulary"]
    )

    log_col_config = {
        "Time": st.column_config.TextColumn("Time", width="small"),
        "Disease": st.column_config.TextColumn("Disease", width="medium"),
        "Conf.": st.column_config.TextColumn("Conf.", width="small"),
        "Severity": st.column_config.TextColumn("Severity", width="small"),
        "Pesticide": st.column_config.TextColumn("Pesticide", width="large"),
        "Dosage": st.column_config.TextColumn("Dosage", width="small"),
        "Pulse": st.column_config.TextColumn("Pulse", width="small"),
        "Cap": st.column_config.TextColumn("Cap", width="small"),
        "Mode": st.column_config.TextColumn("Mode", width="small"),
    }

    # =========================================================================
    # TAB 1: LIVE CONTROL
    # =========================================================================
    with tab_live:
        c_start, c_stop, c_policy = st.columns([0.9, 0.9, 3.2])
        with c_start:
            if st.button("Start Stream", type="primary", width="stretch"):
                st.session_state.camera_running = True
        with c_stop:
            if st.button("Stop / Pump OFF", width="stretch"):
                st.session_state.camera_running = False
        with c_policy:
            pulse_txt = "Adaptive (1.0/1.5/2.0s)" if adaptive_duration else f"Fixed ({fixed_pump_sec:.1f}s)"
            hw_txt = "DRY-RUN" if dry_run_flag else f"{serial_port} @ {int(baud_rate)}"
            st.markdown(
                f'<div class="policy-bar">Policy:&nbsp;<b>Max {int(max_sprays_per_leaf)}/leaf</b>'
                f'&nbsp;&middot;&nbsp;Pulse:&nbsp;<b>{pulse_txt}</b>'
                f'&nbsp;&middot;&nbsp;Cooldown:&nbsp;<b>{cooldown_sec:.1f}s</b>'
                f'&nbsp;&middot;&nbsp;Link:&nbsp;<b>{html.escape(hw_txt)}</b></div>',
                unsafe_allow_html=True,
            )

        error_banner = st.empty()
        telemetry_placeholder = st.empty()

        vid_col, info_col = st.columns([1.45, 1.05])
        with vid_col:
            st.markdown('<div class="section-label">Live V4 Camera Feed</div>', unsafe_allow_html=True)
            frame_placeholder = st.empty()
        with info_col:
            st.markdown('<div class="section-label">ROI Input &amp; Lesion Segmentation</div>', unsafe_allow_html=True)
            roi_pair_placeholder = st.empty()
            prescription_placeholder = st.empty()

        st.markdown('<div class="section-label" style="margin-top:14px;">Spray Actuation Log</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()

        if st.session_state.camera_running:
            cam = LatestFrameCamera(int(cam_index))
            if not cam.opened:
                error_banner.error(f"Could not open Camera Index {cam_index}. Verify DroidCam or switch to Index 0.")
                st.session_state.camera_running = False
            else:
                pump = SafeSerialPump(port=serial_port, baud=int(baud_rate), dry_run=dry_run_flag)
                vision_state = VisionState()
                vision_state.mode = mode_b_val if mode_b_enabled else v4.MODE_A
                vision_state.corr_mode_idx = corr_modes.index(glare_mode)
                vision_state.wb_enabled = wb_enabled

                sm = LeafBoundedSprayStateMachine(
                    confirm_required=confirm_frames,
                    disease_prob_threshold=disease_thresh,
                    default_pump_on_seconds=fixed_pump_sec,
                    cooldown_seconds=cooldown_sec,
                    max_sprays_per_leaf=int(max_sprays_per_leaf),
                    reset_clear_sec=1.0,
                )

                last_log_len = -1
                last_rx_key = None
                last_info = None
                last_rec = get_pesticide_recommendation("Apple___healthy")
                last_sev = {"severity_pct": 0.0, "severity_grade": "None", "recommended_pulse_sec": 0.0, "est_volume_ml": 0.0}
                last_leaf_valid = False
                last_is_diseased = False
                last_area_frac = 0.0
                last_diseased_prob = 0.0
                last_healthy_prob = 0.0
                last_frame_ms = 0.0
                last_pulse_to_use = fixed_pump_sec
                last_vol_to_use = 0.0

                if st.session_state.spray_log:
                    log_placeholder.dataframe(
                        pd.DataFrame(st.session_state.spray_log),
                        width="stretch",
                        hide_index=True,
                        column_config=log_col_config,
                    )
                    last_log_len = len(st.session_state.spray_log)
                else:
                    log_placeholder.caption("No spray events recorded in this session.")
                    last_log_len = 0

                try:
                    while st.session_state.camera_running:
                        ok, frame = cam.read()
                        if not ok or frame is None:
                            time.sleep(0.015)
                            continue

                        now = time.monotonic()

                        # FAST PATH during SPRAYING / COOLDOWN:
                        # Skip heavy CNN inference & lesion segmentation so the timer ticks smoothly at ~20 Hz.
                        if sm.state in ("SPRAYING", "COOLDOWN"):
                            action = sm.tick_timers(now)
                            if action == "TURN_OFF":
                                pump.turn_off("spray pulse complete -> cooldown")

                            view = frame.copy()
                            guide = v4.guide_box(frame.shape)
                            band_px = v4.watermark_band_px(frame.shape[0])
                            v4.draw_overlays(view, guide, band_px, None, None, None)

                            rem = max(0.0, sm.state_until - now)
                            if sm.state == "SPRAYING":
                                hud_lines = [
                                    (f"SPRAYING ACTIVE: {sm.trigger_label}  ({rem:.2f}s / {sm.active_pulse_sec:.1f}s)", v4.RED),
                                    (f"Pesticide: {sm.last_pesticide}", v4.WHITE),
                                    (f"Leaf Sprays: {sm.leaf_spray_count}/{sm.max_sprays_per_leaf} | {pump.status_text}", v4.RED),
                                ]
                            elif sm.state == "COOLDOWN":
                                hud_lines = [
                                    (f"SAFETY COOLDOWN LOCKOUT: {rem:.2f}s remaining", v4.WHITE),
                                    (f"Last Trigger: {sm.trigger_label} | Leaf Sprays: {sm.leaf_spray_count}/{sm.max_sprays_per_leaf}", v4.GRAY_TXT),
                                    (f"Pump OFF | {pump.status_text}", v4.GREEN),
                                ]
                            else:
                                hud_lines = [
                                    (f"STATE: {sm.state} | Leaf Sprays: {sm.leaf_spray_count}/{sm.max_sprays_per_leaf}", v4.GREEN),
                                ]

                            canvas = v4.build_main_canvas(view, hud_lines)
                            frame_placeholder.image(encode_jpeg(canvas, quality=72), width="stretch")

                            telemetry_placeholder.markdown(
                                render_telemetry_html(
                                    sm, now, last_leaf_valid, last_is_diseased, last_area_frac,
                                    last_rec, last_diseased_prob, last_healthy_prob, last_frame_ms,
                                    last_pulse_to_use, last_sev,
                                ),
                                unsafe_allow_html=True,
                            )

                            rx_key = (sm.state, sm.leaf_spray_count, last_leaf_valid, last_is_diseased, last_rec["raw_class"])
                            if rx_key != last_rx_key:
                                last_rx_key = rx_key
                                prescription_placeholder.markdown(
                                    render_prescription_html(
                                        sm, last_leaf_valid, last_is_diseased, last_rec,
                                        last_diseased_prob, last_healthy_prob, last_pulse_to_use,
                                        last_vol_to_use, last_sev, pump.status_text,
                                    ),
                                    unsafe_allow_html=True,
                                )

                            time.sleep(0.04)
                            continue

                        # NORMAL INSPECTION PATH (STANDBY / MAX_REACHED)
                        vision_state.mode = mode_b_val if mode_b_enabled else v4.MODE_A
                        vision_state.corr_mode_idx = corr_modes.index(glare_mode)
                        vision_state.wb_enabled = wb_enabled

                        t_frame = time.perf_counter()
                        info = analyze_frame(frame, vision_state, model, device, idx_to_class, healthy_idx)
                        frame_ms = (time.perf_counter() - t_frame) * 1000.0

                        summary = info["summary"]
                        leaf_valid = bool(info["leaf_valid"])
                        area_frac = float(info["area_fraction"])
                        diseased_prob = float(info["diseased_prob"])
                        healthy_prob = float(summary["healthy_prob"]) if summary is not None else 0.0
                        top1_label = (
                            v4.format_class_name(summary["top1"][0]) if summary is not None else "No Leaf"
                        )

                        is_diseased = leaf_valid and (diseased_prob >= disease_thresh)
                        raw_cls = get_raw_class_from_probs(
                            vision_state.smoothed, idx_to_class, healthy_idx, is_diseased
                        )
                        rec = get_pesticide_recommendation(raw_cls)

                        sev, lesion_overlay = estimate_leaf_severity_and_duration(info["preview"], is_diseased)
                        pulse_to_use = (
                            sev["recommended_pulse_sec"] if (adaptive_duration and is_diseased) else fixed_pump_sec
                        )
                        vol_to_use = round(pulse_to_use * 12.0, 1) if is_diseased else 0.0
                        diseased_display = f"{rec['crop']} - {rec['disease']}"

                        action = sm.step(
                            now=now,
                            leaf_valid=leaf_valid,
                            diseased_prob=diseased_prob,
                            diseased_label=diseased_display if is_diseased else top1_label,
                            pesticide_name=rec["pesticide"],
                            pulse_sec=pulse_to_use,
                            est_volume_ml=vol_to_use,
                        )

                        # Save latest valid snapshot for smooth timer fast-path
                        last_info = info
                        last_rec = rec
                        last_sev = sev
                        last_leaf_valid = leaf_valid
                        last_is_diseased = is_diseased
                        last_area_frac = area_frac
                        last_diseased_prob = diseased_prob
                        last_healthy_prob = healthy_prob
                        last_frame_ms = frame_ms
                        last_pulse_to_use = pulse_to_use
                        last_vol_to_use = vol_to_use

                        if action == "TURN_ON":
                            pump.turn_on(
                                f"{sm.trigger_label} | Spray {sm.leaf_spray_count}/{sm.max_sprays_per_leaf} ({sm.active_pulse_sec:.1f}s)"
                            )
                            st.session_state.spray_log.insert(
                                0,
                                {
                                    "Time": datetime.now().strftime("%H:%M:%S"),
                                    "Disease": sm.trigger_label,
                                    "Conf.": f"{diseased_prob * 100:.1f}%",
                                    "Severity": f"{sev['severity_pct']}% ({sev['severity_grade'].split()[0]})",
                                    "Pesticide": rec["pesticide"],
                                    "Dosage": rec["dosage"],
                                    "Pulse": f"{sm.active_pulse_sec:.1f}s",
                                    "Cap": f"{sm.leaf_spray_count}/{sm.max_sprays_per_leaf}",
                                    "Mode": pump.status_text,
                                },
                            )

                        hud_lines = list(info["lines"])
                        hud_color = v4.WHITE if sm.state == "MAX_REACHED" else v4.GREEN
                        hud_lines.append(
                            (
                                f"PUMP: {sm.state} | Leaf Sprays: {sm.leaf_spray_count}/{sm.max_sprays_per_leaf} | "
                                f"Confirm: {sm.confirm_count}/{sm.confirm_required} | {pump.status_text}",
                                hud_color,
                            )
                        )
                        canvas = v4.build_main_canvas(info["view"], hud_lines)
                        frame_placeholder.image(encode_jpeg(canvas, quality=72), width="stretch")

                        sep = np.full((info["preview"].shape[0], 8, 3), 18, dtype=np.uint8)
                        pair_img = np.hstack([info["preview"], sep, lesion_overlay])
                        roi_pair_placeholder.image(
                            encode_jpeg(pair_img, quality=76),
                            caption=f"LAB-Corrected ROI (224x224)   |   Lesion Severity Mask ({sev['severity_pct']}%)",
                            width="stretch",
                        )

                        telemetry_placeholder.markdown(
                            render_telemetry_html(
                                sm, now, leaf_valid, is_diseased, area_frac, rec,
                                diseased_prob, healthy_prob, frame_ms, pulse_to_use, sev,
                            ),
                            unsafe_allow_html=True,
                        )

                        rx_key = (
                            sm.state,
                            sm.leaf_spray_count,
                            leaf_valid,
                            is_diseased,
                            rec["raw_class"],
                            round(sev["severity_pct"], 0),
                            round(diseased_prob, 2) if is_diseased else round(healthy_prob, 2),
                        )
                        if rx_key != last_rx_key:
                            last_rx_key = rx_key
                            prescription_placeholder.markdown(
                                render_prescription_html(
                                    sm, leaf_valid, is_diseased, rec, diseased_prob,
                                    healthy_prob, pulse_to_use, vol_to_use, sev, pump.status_text,
                                ),
                                unsafe_allow_html=True,
                            )

                        if len(st.session_state.spray_log) != last_log_len:
                            last_log_len = len(st.session_state.spray_log)
                            log_placeholder.dataframe(
                                pd.DataFrame(st.session_state.spray_log),
                                width="stretch",
                                hide_index=True,
                                column_config=log_col_config,
                            )

                        time.sleep(0.02)
                except Exception as exc:
                    error_banner.error(f"Stream runtime error: {exc}")
                finally:
                    pump.close()
                    cam.release()
        else:
            idle_sm = LeafBoundedSprayStateMachine(max_sprays_per_leaf=int(max_sprays_per_leaf))
            telemetry_placeholder.markdown(
                render_telemetry_html(
                    idle_sm,
                    time.monotonic(),
                    False,
                    False,
                    0.0,
                    get_pesticide_recommendation("Apple___healthy"),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    {"severity_pct": 0.0, "severity_grade": "None"},
                ),
                unsafe_allow_html=True,
            )
            prescription_placeholder.markdown(
                render_prescription_html(
                    idle_sm,
                    False,
                    False,
                    get_pesticide_recommendation("Apple___healthy"),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    {"severity_pct": 0.0, "severity_grade": "None"},
                    "DRY-RUN" if dry_run_flag else f"{serial_port} @ {int(baud_rate)}",
                ),
                unsafe_allow_html=True,
            )
            if st.session_state.spray_log:
                log_df = pd.DataFrame(st.session_state.spray_log)
                log_placeholder.dataframe(
                    log_df,
                    width="stretch",
                    hide_index=True,
                    column_config=log_col_config,
                )
                st.download_button(
                    "Export Spray Audit Log (CSV)",
                    data=log_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"pesticide_spray_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                log_placeholder.caption("Press Start Stream to begin real-time inspection and actuation.")

    # =========================================================================
    # TAB 2: IMAGE DIAGNOSTICS
    # =========================================================================
    with tab_upload:
        st.markdown('<div class="section-label">Offline Leaf Diagnostic &amp; Prescription Test</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a leaf image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            resized_bgr = cv2.resize(bgr_img, (v4.IMAGE_SIZE, v4.IMAGE_SIZE))

            full_mask = np.full((v4.IMAGE_SIZE, v4.IMAGE_SIZE), 255, dtype=np.uint8)
            glare_ratio, mean_v = v4.measure_glare(resized_bgr, full_mask)
            should_glare = (glare_mode == "ON") or (
                glare_mode == "AUTO"
                and (glare_ratio > v4.GLARE_CORRECT_RATIO or mean_v > v4.OVEREXPOSED_MEAN_V)
            )

            if should_glare:
                proc_bgr, _ = v4.correct_exposure_lab(resized_bgr, full_mask, wb_enabled=wb_enabled)
            elif wb_enabled:
                proc_bgr, _ = v4.apply_white_balance(resized_bgr, full_mask)
            else:
                proc_bgr = resized_bgr

            probs, infer_ms = v4.predict_probs(proc_bgr, model, device)
            s = v4.summarize(probs, idx_to_class, healthy_idx)
            healthy_prob = float(s["healthy_prob"])
            diseased_prob = float(s["diseased_prob"])

            is_diseased = diseased_prob >= disease_thresh
            raw_cls = get_raw_class_from_probs(probs, idx_to_class, healthy_idx, is_diseased)
            rec = get_pesticide_recommendation(raw_cls)
            sev, lesion_overlay = estimate_leaf_severity_and_duration(proc_bgr, is_diseased)
            pulse_to_use = (
                sev["recommended_pulse_sec"]
                if (adaptive_duration and is_diseased)
                else (fixed_pump_sec if is_diseased else 0.0)
            )
            vol_to_use = round(pulse_to_use * 12.0, 1) if is_diseased else 0.0

            i1, i2, i3 = st.columns(3)
            i1.image(encode_jpeg(resized_bgr, 90), caption="1. Raw Input (224x224)", width="stretch")
            i2.image(
                encode_jpeg(proc_bgr, 90),
                caption=f"2. LAB Exposure Recovery ({'ON' if should_glare else 'OFF'} | Glare {glare_ratio * 100:.1f}%)",
                width="stretch",
            )
            i3.image(
                encode_jpeg(lesion_overlay, 90),
                caption=f"3. Lesion Segmentation ({sev['severity_pct']}% â€” {sev['severity_grade']})",
                width="stretch",
            )

            col_rx, col_probs = st.columns([1.25, 1.0])
            with col_rx:
                dummy_sm = LeafBoundedSprayStateMachine(max_sprays_per_leaf=int(max_sprays_per_leaf))
                st.markdown(
                    render_prescription_html(
                        dummy_sm, True, is_diseased, rec, diseased_prob, healthy_prob,
                        pulse_to_use, vol_to_use, sev, f"OFFLINE ({infer_ms:.2f} ms on {str(device).upper()})",
                    ),
                    unsafe_allow_html=True,
                )

            with col_probs:
                st.markdown('<div class="section-label" style="margin-top:10px;">Top-5 Softmax Distribution</div>', unsafe_allow_html=True)
                top5_idx = np.argsort(probs)[::-1][:5]
                prob_df = pd.DataFrame(
                    {
                        "Class": [v4.format_class_name(idx_to_class[int(i)]) for i in top5_idx],
                        "Confidence (%)": [round(float(probs[int(i)]) * 100.0, 2) for i in top5_idx],
                    }
                )
                st.bar_chart(prob_df.set_index("Class"))
                st.dataframe(prob_df, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 3: MODEL BENCHMARKS
    # =========================================================================
    with tab_bench:
        st.markdown(
            """
            <div class="telemetry-grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="t-card state-standby">
                    <div class="t-label">Best Validation Accuracy</div>
                    <div class="t-value">99.37%</div>
                    <div class="t-sub">Stage-3 Robust MobileNetV2</div>
                </div>
                <div class="t-card">
                    <div class="t-label">Motion Blur Gain (9x9)</div>
                    <div class="t-value">+64.74%</div>
                    <div class="t-sub">29.47% &rarr; 94.21% (n=190)</div>
                </div>
                <div class="t-card">
                    <div class="t-label">Low-Res Gain (1/4x)</div>
                    <div class="t-value">+53.16%</div>
                    <div class="t-sub">41.05% &rarr; 94.21% (n=190)</div>
                </div>
                <div class="t-card">
                    <div class="t-label">Live CUDA Latency</div>
                    <div class="t-value">9.9â€“17.3 ms</div>
                    <div class="t-sub">54,305 SHA-256 audited imgs</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 1. Four-Stage Model Progression")
        prog_df = pd.DataFrame(
            [
                ["best_cnn.pth", "Custom 4-Block CNN (~422k params)", "1.73 MB", "14 / 19", "0.0763", "97.64%", "0.9685", "97.40% (212 err)"],
                ["best_mobilenet_v2.pth", "Stage-1: Frozen Backbone + Head (LR=1e-3)", "9.34 MB", "11 / 16", "0.0822", "97.35%", "0.9653", "â€”"],
                ["best_mobilenet_v2_finetuned.pth", "Stage-2: features[-2:] + Head (LR=1e-5/1e-4)", "9.35 MB", "9 / 12", "0.0485", "98.43%", "0.9789", "98.49% (123 err)"],
                ["best_mobilenet_v2_robust.pth (Selected)", "Stage-3: Full Net (2.27M, LR=1e-5) + Robust Aug", "9.35 MB", "9 / 10", "0.0193", "99.37%", "0.9925", "Held-out untouched"],
            ],
            columns=["Checkpoint", "Strategy", "Size", "Best Epoch", "Val Loss", "Val Acc", "Val F1", "Test Acc"],
        )
        st.dataframe(prog_df, width="stretch", hide_index=True)
        st.image(str(FIGURES_DIR / "model_progression_4stage.png"), width="stretch")

        st.markdown("#### 2. 190-Image Stratified Domain-Shift Stress Suite")
        stress_df = pd.DataFrame(
            [
                ["Original (Clean 224x224)", "97.89%", "0.9788", "99.47%", "0.9947", "+1.58%"],
                ["Motion Blur (9x9 Horizontal)", "29.47%", "0.2726", "94.21%", "0.9433", "+64.74%"],
                ["Low Resolution (1/4x Scale)", "41.05%", "0.3923", "94.21%", "0.9409", "+53.16%"],
                ["Low Light + Sensor Noise", "35.26%", "0.3162", "52.11%", "0.5367", "+16.84%"],
                ["JPEG Compression (Q = 25)", "82.11%", "0.8218", "95.26%", "0.9525", "+13.16%"],
            ],
            columns=["Condition", "Baseline Acc", "Baseline F1", "Robust Acc", "Robust F1", "Acc Gain"],
        )
        st.dataframe(stress_df, width="stretch", hide_index=True)
        st.image(str(FIGURES_DIR / "domain_shift_190_comparison.png"), width="stretch")

        st.markdown("#### 3. Stage-3 Training & Loss Convergence")
        st.image(str(FIGURES_DIR / "stage3_training_curves.png"), width="stretch")

        st.markdown("#### 4. Method Ablation: V1 Baseline vs. V4/V5 Production")
        ablation_df = pd.DataFrame(
            [
                ["Dataset Split", "Random split (21 duplicate pairs leak)", "SHA-256 hash-grouped 70/15/15 split (0 leakage)"],
                ["Spatial Framing", "Full uncropped 640x480 webcam frame", "Top-10% watermark mask + HSV ROI (4% pad, >=5% area)"],
                ["Exposure & Glare", "Raw RGB (specular glare bleaches lesions)", "Conditional LAB L-channel Gamma(1.45) + CLAHE(1.5)"],
                ["Temporal Filter", "Single-frame raw softmax", "5-frame EMA (0.6/0.4) + 3-frame confirmation (>=70%)"],
                ["Actuation Safety", "Fixed pulse, unlimited sprays per leaf", "Adaptive pulse (1.0/1.5/2.0s) + 2-spray cap & 1.0s reset"],
                ["Agronomy Layer", "Disease class label only", "38-class pesticide, active ingredient, dosage & CSV log"],
            ],
            columns=["Dimension", "V1 Baseline", "V4/V5 Production"],
        )
        st.dataframe(ablation_df, width="stretch", hide_index=True)

        corn_err = FIGURES_DIR / "corn_disease_errors.png"
        if corn_err.exists():
            with st.expander("Custom CNN Diagnostic Error Grid (Corn Gray Leaf Spot vs. Northern Leaf Blight)"):
                st.image(str(corn_err), width="stretch")

    # =========================================================================
    # TAB 4: PESTICIDE FORMULARY
    # =========================================================================
    with tab_formulary:
        st.markdown('<div class="section-label">38-Class Crop Pathology &amp; Chemical Formulary</div>', unsafe_allow_html=True)
        search_q = st.text_input("Filter by crop, disease, pathogen, or active ingredient", placeholder="e.g., Apple, Blight, Copper, Tomato")
        rows = [
            {
                "Crop": d["crop"],
                "Condition": d["disease"],
                "Category": d["group"],
                "Pathogen": d["pathogen"],
                "Recommended Pesticide": d["pesticide"],
                "Active Ingredient": d["active_ingredient"],
                "Dosage": d["dosage"],
                "Notes": d["notes"],
            }
            for d in PESTICIDE_DB.values()
        ]
        form_df = pd.DataFrame(rows)
        if search_q.strip():
            q = search_q.strip().lower()
            form_df = form_df[form_df.apply(lambda r: q in " ".join(r.astype(str)).lower(), axis=1)]
        st.dataframe(form_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

