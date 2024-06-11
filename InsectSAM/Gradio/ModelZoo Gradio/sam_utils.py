import os

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import gradio as gr
import json


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult], include_bboxes: bool = True) -> np.ndarray:
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        if include_bboxes:
            color = np.random.randint(0, 256, size=3).tolist()
            cv2.rectangle(image_cv2, (box.xmin, box.ymin),
                          (box.xmax, box.ymax), color, 2)
            cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(image: Union[Image.Image, np.ndarray], detections: List[DetectionResult], include_bboxes: bool = True) -> np.ndarray:
    annotated_image = annotate(image, detections, include_bboxes)
    return annotated_image


def load_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str) and image.startswith("http"):
        image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def get_boxes(detection_results: List[DetectionResult]) -> List[List[List[float]]]:
    boxes = []
    for result in detection_results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([])
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float().permute(0, 2, 3, 1).mean(
        axis=-1).numpy().astype(np.uint8)
    masks = (masks > 0).astype(np.uint8)
    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            masks[idx] = cv2.fillPoly(
                np.zeros(shape, dtype=np.uint8), [polygon], 1)
    return list(masks)


def detect(image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None) -> List[Dict[str, Any]]:
    detector_id = detector_id if detector_id else "IDEA-Research/grounding-dino-base"
    object_detector = pipeline(
        model=detector_id, task="zero-shot-object-detection", device="cpu")
    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = object_detector(
        image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(result) for result in results]


def segment(image: Image.Image, detection_results: List[DetectionResult], polygon_refinement: bool = False, segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    segmenter_id = segmenter_id if segmenter_id else "martintmv/InsectSAM"
    segmentator = AutoModelForMaskGeneration.from_pretrained(
        segmenter_id).to("cpu")
    processor = AutoProcessor.from_pretrained(segmenter_id)
    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes,
                       return_tensors="pt").to("cpu")
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]
    masks = refine_masks(masks, polygon_refinement)
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask
    return detection_results


def grounded_segmentation(image: Union[Image.Image, str], labels: List[str], threshold: float = 0.3, polygon_refinement: bool = False, detector_id: Optional[str] = None, segmenter_id: Optional[str] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
    image = load_image(image)
    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)
    return np.array(image), detections


def mask_to_min_max(mask: np.ndarray) -> Tuple[int, int, int, int]:
    y, x = np.where(mask)
    return x.min(), y.min(), x.max(), y.max()


def extract_and_paste_insect(original_image: np.ndarray, detection: DetectionResult, background: np.ndarray) -> None:
    mask = detection.mask
    xmin, ymin, xmax, ymax = mask_to_min_max(mask)
    insect_crop = original_image[ymin:ymax, xmin:xmax]
    mask_crop = mask[ymin:ymax, xmin:xmax]

    insect = cv2.bitwise_and(insect_crop, insect_crop, mask=mask_crop)

    x_offset, y_offset = xmin, ymin
    x_end, y_end = x_offset + insect.shape[1], y_offset + insect.shape[0]

    insect_area = background[y_offset:y_end, x_offset:x_end]
    insect_area[mask_crop == 1] = insect[mask_crop == 1]


def create_yellow_background_with_insects(image: np.ndarray) -> np.ndarray:
    labels = ["insect"]

    original_image, detections = grounded_segmentation(
        image, labels, threshold=0.3, polygon_refinement=True)

    yellow_background = np.full(
        (original_image.shape[0], original_image.shape[1], 3), (0, 255, 255), dtype=np.uint8)  # BGR for yellow
    for detection in detections:
        if detection.mask is not None:
            extract_and_paste_insect(
                original_image, detection, yellow_background)
    # Convert back to RGB to match Gradio's expected input format
    yellow_background = cv2.cvtColor(yellow_background, cv2.COLOR_BGR2RGB)
    return yellow_background


def run_length_encoding(mask):
    pixels = mask.flatten()
    rle = []
    last_val = 0
    count = 0
    for pixel in pixels:
        if pixel == last_val:
            count += 1
        else:
            if count > 0:
                rle.append(count)
            count = 1
            last_val = pixel
    if count > 0:
        rle.append(count)
    return rle


def detections_to_json(detections):
    detections_list = []
    for detection in detections:
        detection_dict = {
            "score": detection.score,
            "label": detection.label,
            "box": {
                "xmin": detection.box.xmin,
                "ymin": detection.box.ymin,
                "xmax": detection.box.xmax
            },
            "mask": run_length_encoding(detection.mask) if detection.mask is not None else None
        }
        detections_list.append(detection_dict)
    return detections_list


def crop_bounding_boxes_with_yellow_background(image: np.ndarray, yellow_background: np.ndarray, detections: List[DetectionResult]) -> List[np.ndarray]:
    crops = []
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.box.xyxy
        crop = yellow_background[ymin:ymax, xmin:xmax]
        crops.append(crop)
    return crops
