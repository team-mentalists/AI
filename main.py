"""
Live webcam detection using the trained stationary-items model.

Run:
    python main.py

Press Q in the webcam window to quit.
"""

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
CAMERA_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.25


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run 'python train.py' first, or copy your trained best.pt beside main.py."
        )

    model = YOLO(str(MODEL_PATH))
    device = 0 if torch.cuda.is_available() else "cpu"

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Check that it is connected, not being used "
            "by another application, and that camera permission is enabled."
        )

    print("Webcam started. Press Q in the video window to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=device,
                verbose=False,
            )

            annotated = results[0].plot()
            cv2.imshow("Stationary Items Detector", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
