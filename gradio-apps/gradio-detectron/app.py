import gradio as gr
import json
import numpy as np
from detectron_utils import detectron_process_image
def process_image(image):
    detectron_result=detectron_process_image(image)


    return  detectron_result

examples = [
    ["demo.jpg"]
]

gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil")],
    outputs=[
             gr.Image(label='Detectron', type="numpy")],
    title="RB-IBDM Detectron2 Demo 🐞",
    examples=examples
).launch()
