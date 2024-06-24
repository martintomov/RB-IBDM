---
sidebar_position: 1
---
# Usage

## Overview
This notebook is designed to detect insects in images, segment them, and replace the background with a yellow screen. It leverages the Grounding DINO, Segment Anything Model (SAM), and LaMa for achieving these tasks. Follow the instructions below to use the notebook and process your own images.

<a href="https://colab.research.google.com/drive/1Jo2wTp4dDEPa-8GOz91bJoJBdkyLhH2J?usp=share_link" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### ⚠️ Disclaimer

**This page covers the basic process workflow. The provided notebook gives the additional option of running GSL on either a single image or a folder of images and saves the results by default.**


## Prerequisites
1. **Google Colab**: Ensure you are running this notebook in Google Colab for seamless execution of the code, especially for GPU support.
2. **Drive Access**: Mount your Google Drive to access and store files.

## Steps to Use

### 1. Mount Google Drive
Mount your Google Drive to access and save images.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Environment Requirements
Install the necessary libraries and clone the required repositories.

```python
# GroundedSAM
%cd /content
!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
%cd /content/Grounded-Segment-Anything
!pip install -q -r requirements.txt
%cd /content/Grounded-Segment-Anything/GroundingDINO
!pip install -q .
%cd /content/Grounded-Segment-Anything/segment_anything
!pip install -q .
%cd /content/Grounded-Segment-Anything

# LaMa
!pip install -q simple-lama-inpainting
```

### 3. Add Necessary Imports
Import the required libraries and modules for the process.

```python
import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
from torchvision.ops import box_convert

# GroundingDINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from huggingface_hub import hf_hub_download

import supervision as sv

# SAM
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# LaMa
import requests
import torch
from io import BytesIO
from simple_lama_inpainting import SimpleLama
```

### 4. Load and Preprocess Image
Load the image you want to process from your Google Drive.

```python
# Load image
local_image_path = '/content/drive/MyDrive/path_to_your_image.jpg'
image_source, image = load_image(local_image_path)
```

### 5. Detect and Segment Insects
Use Grounding DINO and SAM to detect and segment insects in the image.

```python
annotated_frame, detected_boxes, phrases = detect(image, model=groundingdino_model)
segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes[indices])
```

### 6. Dilate mask and InPaint selection
Use LaMa to inpaint the masked insects in order to effectively remove them from the image.

```python
mask = dilate_mask(mask)
result = simple_lama(image_source, dilated_image_mask_pil)
```

### 7. Replace Background
Create a mask through the extraction of the insects by subtracting the original and inpainted images and replace the background by overlaying the masked original image over a yellow screen.

```python
# Subtract original and inpainted images
img1 = Image.fromarray(image_source)
img2 = result
diff = ImageChops.difference(img2, img1)

# Create mask
threshold = 7
diff2 = diff.convert('L')
diff2 = diff2.point( lambda p: 255 if p > threshold else 0 )
diff2 = diff2.convert('1')
 # Create Yellow background
img3 = Image.new('RGB', img1.size, (255, 236, 10))
diff3 = Image.composite(img1, img3, diff2)
```

### 7. Save the Processed Image
Save the processed image back to your Google Drive.

```python
output_path = '/content/drive/MyDrive/processed_image.jpg'
diff3.save(output_path)
```

## Conclusion
By following these steps, you can use the notebook to process images, detect and segment insects, and replace the background with a yellow screen. Adjust the file paths and parameters as needed to fit your specific use case.
