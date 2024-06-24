import gradio as gr
import json
import numpy as np
from yolo_utils import yolo_processimage
def process_image(image):
    yolo_result = yolo_processimage(image)

    return  yolo_result

examples = [
    ["demo.jpg"]
]

gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil")],
    outputs=[gr.Image(label='Yolov8', type="numpy"),],
    title="RB-IBDM Yolo Demo 🐞",
    examples=examples
).launch()
