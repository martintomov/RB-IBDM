try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

import gradio as gr
import json
import numpy as np
from sam_utils import grounded_segmentation, create_yellow_background_with_insects
from yolo_utils import yolo_processimage
from detectron_utils import detectron_process_image
def process_image(image, include_json):
    detectron_result=detectron_process_image(image)
    yolo_result = yolo_processimage(image)
    insectsam_result = create_yellow_background_with_insects(image)

    return insectsam_result, yolo_result, detectron_result

examples = [
    ["demo.jpg"],
    ["demo1.jpg"],
]

gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil")],
    outputs=[gr.Image(label='InsectSAM', type="numpy"),
             gr.Image(label='Yolov8', type="numpy"),
             gr.Image(label='Detectron', type="numpy")],
    title="Insect Model Zoo 🐞🔬",
    examples=examples
).launch()
