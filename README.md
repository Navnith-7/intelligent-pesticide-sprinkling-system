Intelligent Pesticide Sprinkling System Using Domain-Robust CNN & OpenCV V4

An end-to-end computer vision and embedded hardware system that detects plant leaf diseases in real time (~11–17 ms GPU inference latency) across 38 crop-disease classes and actuates an Arduino relay-controlled pesticide pump using a multi-frame verification state machine.

Unlike standard PlantVillage classifiers that fail under real-world camera domain shift (motion blur, monitor/sunlight glare, low resolution, and background shortcut learning), this project introduces a Domain-Robust MobileNetV2 checkpoint paired with the V4 Dynamic ROI + LAB-Space Highlight Recovery Pipeline.

Key Engineering Highlights

High Baseline & Fine-Tuned Accuracy: Trained via two-phase transfer learning on the PlantVillage dataset (54,305 images across 38 classes), achieving 99.37% validation accuracy and 0.9925 Macro-F1 across 8,146 validation images.

Domain-Shift Stress Testing (+64.74% Gain): Evaluated across a 190-image synthetic real-world stress suite (5 corruption types $\times$ 3 severity levels), boosting motion-blur accuracy from 29.47% $\rightarrow$ 94.21% and low-resolution accuracy from 41.05% $\rightarrow$ 94.21%.

V4 Dynamic ROI & LAB Glare Suppression: Eliminates background shortcut learning and camera overexposure via HSV contour centroid localization, tight 4%-padded bounding-box extraction, and automatic LAB $L$-channel gamma compression ($\gamma = 1.45$) + CLAHE to recover washed-out lesions without distorting leaf chrominance ($A/B$ channels).

Hardware Safety State Machine: Prevents accidental pesticide discharge using a 5% minimum leaf-area gate, 5-frame EMA probability smoothing, a 3-consecutive-frame confirmation counter ($\ge 70\%$ disease probability), and a 2.0-second post-spray hardware safety cooldown.

System Architecture

+-----------------------------------------------------------------------------------+
|                        1. LIVE CAMERA ACQUISITION (DroidCam / USB)                |
|            640x480 @ 30 FPS  |  Top 10% Watermark Exclusion Zone                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     2. V4 OPENCV SPATIAL ISOLATION & ROI CROP                     |
|  * HSV Leaf Segmentation ([18, 25, 25] -> [95, 255, 255]) + Morphological Close   |
|  * Guardrail: Largest Contour Area >= 5.0% of Frame & Centroid in Guide Box       |
|  * Tight Square Bounding Box (4% Pad) -> Mode A (Natural) or Mode B (Feathered)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   3. ADAPTIVE OPTICS & LAB HIGHLIGHT RECOVERY                     |
|  * Glare Detector: Triggers if Glare Ratio (V > 235) > 5% or Mean V > 175         |
|  * LAB Color Space Conversion: Applies Gamma (1.45) + CLAHE (clip=1.5) strictly   |
|    to L-Channel, preserving true olive/brown lesion hues in A & B channels        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               4. DOMAIN-ROBUST MOBILENETV2 INFERENCE (~11-17 ms CUDA)             |
|  * 38-Class Softmax Output  |  5-Frame Exponential Moving Average (0.6 / 0.4)     |
|  * Binary Aggregation: Sum(Healthy Classes) vs. Sum(Diseased Classes)             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                 5. ARDUINO RELAY & PUMP SAFETY STATE MACHINE                      |
|  * Requires 3 Consecutive Confirmed Disease Frames (Diseased Prob >= 70%)         |
|  * Serial Trigger ('1' / 'S') -> 2.0s Hardware Cooldown -> Auto Safe Reset ('0')  |
+-----------------------------------------------------------------------------------+


1. Validation & Domain-Robustness Benchmarks

Clean Validation Performance (8,146 Images, 38 Classes)

Model Checkpoint

Validation Accuracy

Macro Precision

Macro Recall

Macro F1-Score

Inference Latency (RTX GPU)

Baseline MobileNetV2 (best_mobilenet_v2.pth)

99.37%

0.9930

0.9922

0.9925

11.4 ms

Domain-Robust MobileNetV2 (best_mobilenet_v2_robust.pth)

98.95%

0.9888

0.9881

0.9884

11.2 ms

Real-World Corruption Stress Benchmark (190-Image Evaluation Suite)

Both checkpoints were evaluated across severe camera degradations (Level 3 severity shown below; full multi-level reports available in reports/results/):

Corruption Type (Severity Level 3)

Baseline Model Accuracy

Domain-Robust Model Accuracy

Absolute Robustness Gain

Clean Images (Control)

98.95%

98.95%

0.00%

Motion Blur (Camera Shake)

29.47%

94.21%

+64.74%

Low Resolution (Distant Leaf Crop)

41.05%

94.21%

+53.16%

Gaussian Blur (Out-of-Focus Optics)

51.05%

95.26%

+44.21%

Gaussian Sensor Noise (High ISO)

48.95%

65.79%

+16.84%

Extreme Brightness / Glare Shift

79.47%

85.79% (+ V4 LAB Recovery)

+6.32%

2. Live V4 Vision & Hardware Verification

Test Scenario

Pipeline Mode

Top-1 Prediction & Confidence

Aggregate Decision

Inference Time

Hardware State

Apple — Healthy

Mode A (Natural ROI)

Apple - healthy (99.4%)

HEALTHY (99.4%)

11.4 ms

STANDBY (0/3) — NO SPRAY

Apple — Apple Scab

Mode A + LAB Glare On

Apple - Apple scab (95.7%)

DISEASED (95.7%)

11.2 ms

SPRAY READY

Strawberry — Healthy

Mode A (Natural ROI)

Strawberry - healthy (96.8%)

HEALTHY (97.7%)

14.5 ms

STANDBY (0/3) — NO SPRAY

Strawberry — Healthy

Mode B (Feathered Gray)

Strawberry - healthy (76.9%)

HEALTHY (98.4%)

24.4 ms

STANDBY (0/3) — NO SPRAY

Strawberry — Leaf Scorch

Mode A (Natural ROI)

Strawberry - Leaf scorch (100.0%)

DISEASED (100.0%)

17.2 ms

SPRAYING! $\rightarrow$ 2.0s COOLDOWN

3. Repository Structure

intelligent-pesticide-sprinkling-system/
├── models/
│   ├── best_mobilenet_v2.pth                 # Clean baseline checkpoint (99.37% val acc)
│   └── best_mobilenet_v2_robust.pth          # Domain-robust checkpoint (+64.74% blur gain)
├── reports/
│   └── results/
│       ├── robust_model_190_summary.csv      # Baseline vs. Robust stress-test summary
│       └── robust_model_190_detailed.csv     # Per-image corruption benchmark logs
├── src/
│   ├── data/
│   │   └── dataset.py                        # PlantVillage 38-class dataset & transforms
│   ├── hardware/
│   │   └── arduino_controller.py             # Serial relay interface & fail-safe reset
│   ├── models/
│   │   ├── mobilenet.py                      # MobileNetV2 architecture definition
│   │   └── finetune_mobilenet_robust.py      # Domain-robust augmentation & fine-tuning
│   └── inference/
│       ├── live_sprinkler_v4.py              # MAIN PRODUCTION SCRIPT (Vision V4 + Arduino)
│       ├── live_predict_v4.py                # V4 Vision Debugger with 224x224 CNN preview
│       ├── evaluate_robust_model.py          # 190-image multi-severity benchmark runner
│       ├── live_predict.py                   # Baseline uncropped camera inference script
│       └── predict.py                        # Single-image CLI inference utility
├── requirements.txt
└── README.md


4. Quick-Start & Execution Guide

Environment Setup

git clone https://github.com/Navnith-7/intelligent-pesticide-sprinkling-system.git
cd intelligent-pesticide-sprinkling-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


A. Run Full Production System (Simulated / Dry-Run Mode)

Use --dry-run to run the complete V4 vision pipeline and state machine on screen without requiring the physical Arduino over USB:

.\.venv\Scripts\python.exe -m src.inference.live_sprinkler_v4 --dry-run


B. Run Full Production System with Arduino Hardware

Connect your Arduino relay controller over USB, verify the COM port in Device Manager, and run:

.\.venv\Scripts\python.exe -m src.inference.live_sprinkler_v4 --port COM3


C. Run Standalone V4 Vision Debugger (with 224×224 Crop Inspector)

.\.venv\Scripts\python.exe -m src.inference.live_predict_v4


D. Reproduce the 190-Image Robustness Benchmark

.\.venv\Scripts\python.exe -m src.inference.evaluate_robust_model


5. Interactive Runtime Controls (HUD Hotkeys)

While live_sprinkler_v4.py or live_predict_v4.py is running, use these keyboard shortcuts:

Key

Function

Description

m

Toggle ROI Mode (Mode A / Mode B)

Switches between Mode A (Natural ROI Crop) for maximum lesion edge accuracy and Mode B (Feathered Neutral-Gray Mask) for cluttered backgrounds.

c

Cycle LAB Glare Recovery

Cycles between AUTO (triggers when Glare > 5% or mean V > 175), ON, and OFF.

w

Toggle Background White Balance

Toggles Gray-World background color correction (OFF by default to preserve natural warm leaf/background tones).

q / Esc

Safe Shutdown

Immediately sends pump OFF command, releases serial port, and closes OpenCV windows.

Author

Navnith N Reddy

B.E. in Artificial Intelligence & Machine Learning — New Horizon College of Engineering, Bengaluru

GitHub | LinkedIn | Email