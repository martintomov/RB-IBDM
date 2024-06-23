---
sidebar_position: 1
---
# Usage

## Overview
This notebook uses Detectron2's Faster R-CNN model to detect and classify insects in images. It guides you through setting up the environment, loading the model, processing images, and visualizing the results.

## Prerequisites
1. **Google Colab**: Ensure you are running this notebook in Google Colab for GPU support.
2. **Drive Access**: Mount your Google Drive to access and store files.

## Steps to Use

### 1. Mount Google Drive
Mount your Google Drive to access and save images.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Detectron2
Install Detectron2 library and its dependencies.

```python
!pip install -U torch torchvision
!pip install cython pyyaml==5.1
!pip install 'git+https://github.com/facebookresearch/fvcore'
!pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

### 3. Add Necessary Imports
Import the required libraries and modules for the process.

```python
import torch, torchvision
import cv2
import random
import numpy as np
import os
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
```

### 4. Load and Preprocess Image
Load the image you want to process from your Google Drive.

```python
image_path = '/content/drive/MyDrive/path_to_your_image.jpg'
image = cv2.imread(image_path)
cv2_imshow(image)
```

### 5. Configure the Model
Set up the configuration for the Detectron2 Faster R-CNN model.

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

### 6. Detect and Classify Insects
Use the model to detect and classify insects in the image.

```python
outputs = predictor(image)
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])
```

### 7. Save the Processed Image
Save the processed image back to your Google Drive.

```python
output_path = '/content/drive/MyDrive/processed_image.jpg'
cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
```

## Conclusion
By following these steps, you can use the notebook to process images, detect and classify insects using Detectron2's Faster R-CNN model, and visualize the results. Adjust the file paths and parameters as needed to fit your specific use case.