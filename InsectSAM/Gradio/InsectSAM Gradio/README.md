# InsectSAM 🐞

![InsectSAM Hugging Face App](https://i.imgur.com/cCgpvx9.png)

![Demo](/InsectSAM/Gradio/video-demos/demo-gif.gif)

## Insect Detection and Dataset Preparation for DIOPSIS and ARISE Algorithms

This application facilitates the preprocessing of images for subsequent analysis using the DIOPSIS and ARISE algorithms, focusing on insect biodiversity detection. It leverages advanced deep learning models, including the InsectSAM model developed by Fontys UAS Eindhoven's team RB-IBDM and the GroundingDINO model by IDEA Research. By annotating images with bounding boxes, labels, and masks, this preprocessing step enhances the suitability of images for running on the DIOPSIS and ARISE algorithms.

## Description

The primary goal of this application is to prepare images with insects for further analysis using the DIOPSIS and ARISE algorithms. These algorithms are essential for detecting and analyzing insect biodiversity, but they require images that are appropriately preprocessed to ensure accurate results.

To achieve this, the application employs deep learning models for insect detection and segmentation. The InsectSAM model is utilized to segment insects within images, while the GroundingDINO model enables zero-shot object detection of insects. By combining these models, the application can accurately identify insects within images, even in diverse and complex backgrounds to which insects are naturally attracted.

Furthermore, the application generates annotations such as bounding boxes and masks, providing valuable information about the location and shape of detected insects. These preprocessing steps are crucial for the subsequent analysis performed by the DIOPSIS and ARISE algorithms, ensuring that they can effectively identify and quantify insect biodiversity.

## Installation

To use this application, follow these steps:

1. Clone this repository to your local machine.
2. Create a conda environment for the project.
3. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage

Once the installation is complete, you can run the application by executing the `app.py` script:

```bash
python app.py
```

This will launch a Gradio interface where you can upload images and visualize the detected insects along with the generated annotations.

## Acknowledgments

- **InsectSAM**: Developed by Fontys UAS Eindhoven's team RB-IBDM, the InsectSAM model is utilized for insect segmentation.
- **GroundingDINO**: Developed by IDEA Research, GroundingDINO is employed for zero-shot object detection of objects.
- **DIOPSIS and ARISE Algorithms**: These algorithms are utilized for insect biodiversity detection.

## References

- [InsectSAM Hugging Face App](https://huggingface.co/spaces/martintmv/InsectSAM)
- [InsectSAM Hugging Face Repository](https://huggingface.co/martintmv/InsectSAM)
- [GroundingDINO Hugging Face Repository](https://huggingface.co/IDEA-Research/grounding-dino-base)

For more details on the models and algorithms used in this application, please refer to the respective repositories.

---

**Disclaimer**: This application is for demonstration purposes only and should not be used for critical insect detection tasks without proper evaluation and validation.
