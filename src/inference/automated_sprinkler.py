
from pathlib import Path
import sys
import time

import cv2
import serial
from PIL import Image

# Add project root for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PlantDiseasePredictor


# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_MSMF

ARDUINO_PORT = "COM6"
BAUD_RATE = 9600

PREDICTION_INTERVAL = 1.0
CONFIDENCE_THRESHOLD = 0.90

CONFIRMATION_FRAMES = 3
CLEAR_FRAMES_TO_REARM = 3

MAX_PUMP_ON_SECONDS = 1.5
COOLDOWN_SECONDS = 10.0

WINDOW_NAME = "Intelligent Pesticide Sprinkling System"


# ---------------- HELPER FUNCTIONS ----------------

def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


def is_healthy(class_name):
    return class_name.lower().endswith("healthy")


def send_command(arduino, command):
    if command not in ("1", "0"):
        raise ValueError("Invalid Arduino command")

    arduino.write(command.encode("ascii"))
    arduino.flush()


# ---------------- MAIN PROGRAM ----------------

def main():

    arduino = None
    camera = None
    pump_is_on = False

    try:
        # Connect to Arduino.
        print("Connecting to Arduino...")

        arduino = serial.Serial(
            ARDUINO_PORT,
            BAUD_RATE,
            timeout=1
        )

        # Arduino resets when serial connection opens.
        time.sleep(2)
        arduino.reset_input_buffer()

        send_command(arduino, "0")

        print("Arduino connected.")
        print("Pump initialized to OFF.")

        # Load CNN.
        print("\nLoading MobileNetV2...")
        predictor = PlantDiseasePredictor()

        # Open DroidCam.
        print("\nOpening DroidCam...")

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            CAMERA_BACKEND
        )

        if not camera.isOpened():
            raise RuntimeError("Could not open DroidCam.")

        print("DroidCam connected.")
        print("\nAutomated sprinkler system ready.")
        print("Press Q or ESC to exit.\n")

        # Prediction state.
        last_prediction_time = 0.0
        last_result = None
        prediction_error = None

        previous_class = None
        consecutive_frames = 0
        clear_frames = 0

        # Pump state.
        pump_on_until = 0.0
        last_activation_time = -COOLDOWN_SECONDS
        armed = True

        while True:

            success, frame = camera.read()

            if not success or frame is None:
                raise RuntimeError("Camera frame could not be read.")

            current_time = time.monotonic()

            # Turn pump OFF after its maximum activation duration.
            if pump_is_on and current_time >= pump_on_until:
                send_command(arduino, "0")
                pump_is_on = False
                print("Pump automatically switched OFF.")

            # Run CNN prediction at the configured interval.
            if current_time - last_prediction_time >= PREDICTION_INTERVAL:

                try:
                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    pil_image = Image.fromarray(rgb_frame)

                    last_result = predictor.predict_image(pil_image)
                    prediction_error = None

                except Exception as error:
                    prediction_error = str(error)

                    previous_class = None
                    consecutive_frames = 0
                    clear_frames = 0

                    # Fail safe on prediction errors.
                    if pump_is_on:
                        send_command(arduino, "0")
                        pump_is_on = False

                    print("Prediction error:", error)

                last_prediction_time = time.monotonic()

                if last_result is not None and not prediction_error:

                    predicted_class = last_result["predicted_class"]
                    confidence = last_result["confidence"]

                    healthy = is_healthy(predicted_class)

                    valid_prediction = (
                        confidence >= CONFIDENCE_THRESHOLD
                        and not healthy
                    )

                    # Count consecutive matching disease predictions.
                    if valid_prediction:

                        clear_frames = 0

                        if predicted_class == previous_class:
                            consecutive_frames += 1
                        else:
                            previous_class = predicted_class
                            consecutive_frames = 1

                    else:
                        previous_class = None
                        consecutive_frames = 0
                        clear_frames += 1

                        # Require several clear frames before rearming.
                        if clear_frames >= CLEAR_FRAMES_TO_REARM:
                            armed = True

                    cooldown_finished = (
                        time.monotonic() - last_activation_time
                        >= COOLDOWN_SECONDS
                    )

                    should_activate = (
                        armed
                        and not pump_is_on
                        and cooldown_finished
                        and valid_prediction
                        and consecutive_frames >= CONFIRMATION_FRAMES
                    )

                    if should_activate:

                        print("\nDisease confirmed:", predicted_class)
                        print(f"Confidence: {confidence:.1%}")
                        print("Activating water pump...")

                        send_command(arduino, "1")

                        pump_is_on = True
                        pump_on_until = (
                            time.monotonic() + MAX_PUMP_ON_SECONDS
                        )

                        last_activation_time = time.monotonic()
                        armed = False

                        print("Pump ON for a maximum of 1.5 seconds.")

            # Display information.
            if prediction_error:
                lines = [
                    "PREDICTION ERROR",
                    prediction_error[:55],
                    "PUMP: OFF"
                ]

            elif last_result is None:
                lines = [
                    "Initializing...",
                    "PUMP: OFF"
                ]

            else:
                class_name = format_class_name(
                    last_result["predicted_class"]
                )

                confidence = last_result["confidence"]
                inference_ms = last_result["inference_ms"]

                lines = [
                    class_name,
                    f"Confidence: {confidence:.1%}",
                    f"Inference: {inference_ms:.0f} ms",
                    f"Confirmation: {consecutive_frames}/{CONFIRMATION_FRAMES}",
                    f"Pump: {'ON' if pump_is_on else 'OFF'}",
                    f"Armed: {'YES' if armed else 'NO'}"
                ]

            y = 35

            for index, line in enumerate(lines):

                color = (
                    (0, 255, 0)
                    if index == 0
                    else (255, 255, 255)
                )

                cv2.putText(
                    frame,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA
                )

                y += 35

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except Exception as error:
        print("\nSYSTEM ERROR:", error)

    finally:
        # Always attempt to switch the pump OFF.
        if arduino is not None and arduino.is_open:
            try:
                send_command(arduino, "0")
                print("Safety command sent: PUMP OFF.")
            except Exception as error:
                print("Could not send OFF command:", error)

            arduino.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("System shutdown complete.")


if __name__ == "__main__":
    main()
