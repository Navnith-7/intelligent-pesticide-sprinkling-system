
import cv2
from pygrabber.dshow_graph import FilterGraph


def main():
    devices = FilterGraph().get_input_devices()

    print("Available cameras:")
    for index, name in enumerate(devices):
        print(f"  {index}: {name}")

    matches = [
        index
        for index, name in enumerate(devices)
        if "DroidCam Video" in name
    ]

    if not matches:
        print("DroidCam Video was not found.")
        return

    camera_index = matches[0]
    print(f"\nOpening DroidCam Video at index {camera_index}...")

    # Media Foundation works with this virtual camera.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Could not open DroidCam with Media Foundation.")
        cap.release()
        return

    window_name = "DroidCam Phone Camera"
    print("Press Q or ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Could not read a frame.")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Camera released. Test stopped.")


if __name__ == "__main__":
    main()
