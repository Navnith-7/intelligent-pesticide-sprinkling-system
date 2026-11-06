
from pathlib import Path
import sys
import time

import cv2
from PIL import Image

# Add the project root so Python can import the src package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor


CAMERA_INDEX = 1
PREDICTION_INTERVAL_SECONDS = 1.0
CONFIDENCE_THRESHOLD = 0.80


def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


def main():
    print("Initializing plant disease detection...")
    predictor = PlantDiseasePredictor()

    print("\nOpening DroidCam...")
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    if not camera.isOpened():
        print("ERROR: Could not open DroidCam.")
        camera.release()
        return

    print("DroidCam connected successfully.")
    print("Show a leaf to the camera.")
    print("Press Q or ESC to exit.\n")

    last_prediction_time = 0.0
    last_result = None
    prediction_error = None
    window_name = "Plant Disease Detection - MobileNetV2"

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("WARNING: Could not read a camera frame.")
                break

            current_time = time.perf_counter()

            if current_time - last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    last_result = predictor.predict_image(pil_image)
                    prediction_error = None
                except Exception as error:
                    prediction_error = str(error)

                last_prediction_time = time.perf_counter()

            if prediction_error:
                lines = ["Prediction error", prediction_error[:65]]
            elif last_result is None:
                lines = ["Initializing prediction..."]
            else:
                class_name = format_class_name(
                    last_result["predicted_class"]
                )
                confidence = last_result["confidence"]
                inference_ms = last_result["inference_ms"]

                status = (
                    "Prediction: HIGH CONFIDENCE"
                    if confidence >= CONFIDENCE_THRESHOLD
                    else "Prediction: UNCERTAIN"
                )

                lines = [
                    class_name,
                    f"Confidence: {confidence:.1%}",
                    f"Inference: {inference_ms:.0f} ms",
                    status,
                ]

            y = 35
            for index, line in enumerate(lines):
                text_size = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )[0]

                right = min(frame.shape[1] - 5, text_size[0] + 25)
                cv2.rectangle(
                    frame, (10, y - 25), (right, y + 8),
                    (25, 25, 25), -1
                )
                cv2.putText(
                    frame, line, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if index == 0 else (255, 255, 255),
                    2, cv2.LINE_AA
                )
                y += 38

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Camera released. Live prediction stopped.")


if __name__ == "__main__":
    main()
