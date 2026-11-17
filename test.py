"""
Test the trained stationary-items model using image upload or webcam.

Run:
    python test.py

Open the local Gradio URL printed in the terminal.
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
IMAGE_SIZE = 640


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run 'python train.py' first, or place your trained best.pt beside test.py."
    )

model = YOLO(str(MODEL_PATH))
print("Model loaded successfully!")
print("Classes:", model.names)


def detect_objects(input_image, confidence_threshold):
    if input_image is None:
        return None

    # Gradio supplies an RGB NumPy image.
    image_rgb = np.asarray(input_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=image_bgr,
        conf=float(confidence_threshold),
        imgsz=IMAGE_SIZE,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        verbose=False,
    )

    annotated_bgr = results[0].plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


web_app = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(
            type="numpy",
            sources=["upload", "webcam"],
            label="Upload an image or capture from webcam",
        ),
        gr.Slider(
            minimum=0.10,
            maximum=1.00,
            value=0.25,
            step=0.05,
            label="Confidence threshold",
        ),
    ],
    outputs=gr.Image(type="numpy", label="Detected stationary items"),
    title="Stationary Items Detector",
    description="Detect tape dispensers, highlighters, glue sticks, pens, scissors, protractors and correction tape.",
)

if __name__ == "__main__":
    web_app.launch(
        share=False,
        show_error=True,
    )
