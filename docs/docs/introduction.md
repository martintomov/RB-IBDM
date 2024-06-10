---
sidebar_position: 1
---
# Introduction

Welcome to the **InsectSAM** documentation!

InsectSAM is an open-source ML model designed for semantic segmentation of insect images. Leveraging advanced deep learning techniques, InsectSAM is capable of accurately identifying and segmenting various insect species in images, making it an invaluable tool for researchers, entomologists, and AI enthusiasts working in the field of insect identification and classification.

![Inference 1](../static/img/demo-intro.png)

## Getting Started

Let's get you started with InsectSAM and guide you through the initial steps to utilize this model effectively.

### What you'll need

- A modern development environment set up with the following:
  - [Python 3.11 or above](https://www.python.org/downloads/)
  - [Hugging Face Transformers library](https://huggingface.co/transformers/installation.html)
  - A compatible GPU setup (optional but recommended for faster processing)

## Installation

To begin using InsectSAM, you'll need to install the necessary dependencies and download the model from Hugging Face. Follow these steps:

1. Install the Hugging Face Transformers library:
   ```bash
   pip install transformers
   ```

2. Download and set up the InsectSAM model:
   ```python
   from transformers import AutoModelForImageSegmentation, AutoProcessor

   model = AutoModelForImageSegmentation.from_pretrained("martintmv/InsectSAM")
   processor = AutoProcessor.from_pretrained("martintmv/InsectSAM")
   ```

## Further Reading

To dive deeper into the capabilities and advanced usage of InsectSAM, refer to the following resources:

- [InsectSAM Model Page on Hugging Face](https://huggingface.co/martintmv/InsectSAM)
- [InsectSAM Demo App on Hugging Face](https://huggingface.co/spaces/martintmv/InsectSAM)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)