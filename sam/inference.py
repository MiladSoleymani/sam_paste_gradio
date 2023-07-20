import sys
import os

sys.path.append(os.getcwd())
sys.path.append("..")

from segment_anything import sam_model_registry, SamPredictor

import cv2
import torch
import numpy as np
from PIL import Image

import argparse

from typing import Dict

import supervision as sv


def script_run(conf: Dict) -> None:
    image = cv2.imread(conf["input_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    sam = sam_model_registry[conf["model_type"]](checkpoint=conf["model_path"])
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[314, 892], [700, 967]])
    input_label = np.array([1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    mask_annotator = sv.MaskAnnotator()
    mask_detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
    mask_detections = mask_detections[
        mask_detections.area == np.max(mask_detections.area)
    ]

    segmented_image = mask_annotator.annotate(
        scene=image.copy(), detections=mask_detections
    )

    cv2.imwrite(conf["save_path"], segmented_image)


class SamGradioRun(object):
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.__device = "cuda"
        else:
            self.__device = "cpu"

    def initialize(self, image, model_type):
        if model_type == "vit_h":
            model_path = "/home/ubuntu/milad/gradio/sam/models/sam_vit_h_4b8939.pth"

        elif model_type == "vit_b":
            model_path = "/home/ubuntu/milad/gradio/sam/models/sam_vit_b_01ec64.pth"

        elif model_type == "vit_l":
            model_path = "/home/ubuntu/milad/gradio/sam/models/sam_vit_l_0b3195.pth"

        self.__real_image = np.copy(image)

        print(f"{model_path = }")
        print(f"{self.__real_image.shape = }")

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.__device)
        self.__predictor = SamPredictor(sam)
        self.__predictor.set_image(image)

    def detect(self, points):
        masks, scores, _ = self.__predictor.predict(
            point_coords=points.get_point_list(),
            point_labels=points.get_label_list(),
            multimask_output=False,
        )

        mask_annotator = sv.MaskAnnotator()
        mask_detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        mask_detections = mask_detections[
            mask_detections.area == np.max(mask_detections.area)
        ]

        segmented_image = mask_annotator.annotate(
            scene=self.__real_image.copy(), detections=mask_detections
        )

        return segmented_image

    def remove_background(self, points):
        masks, scores, _ = self.__predictor.predict(
            point_coords=points.get_point_list(),
            point_labels=points.get_label_list(),
            multimask_output=False,
        )

        best_mask = masks[np.argmax(scores), :, :]
        best_mask = np.repeat(best_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        removed_background = np.array(self.__real_image.copy()) * best_mask
        print(removed_background.shape)
        return Image.fromarray(removed_background.astype(np.uint8))


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/ubuntu/milad/gradio/sam/dataset/input/2023-07-09 01.12.24.jpg",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_b", "vit_l"],
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/milad/gradio/sam/models/sam_vit_h_4b8939.pth",
        help="pretrained sam model path (chose by model type)",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/ubuntu/milad/gradio/sam/dataset/results/results.jpg",
        help="save path folder",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    script_run(conf)
