from abc import ABC
import numpy as np
from typing import Tuple, List
import torch
import os
import cv2
from data.training_sample import TrainingSample, ModelInput, ModelTarget


class RealDataGenerator(ABC):
    def __init__(self):
        pass

    def generate(self, num_samples: int, resize: bool, resize_dims: Tuple[int, int]) -> List[TrainingSample]:
        """If resize, resize_dims must be provided"""
        training_samples = []
        masks_dir = "masks"
        image_dir = "images"
        # print(os.walk(image_dir))
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for box in dirs:
                if "box" in box:
                    image_folder = os.path.join(root, box)
                    mask_folder = os.path.join(masks_dir, box)
                    # identify empty mask
                    for mask_file in os.listdir(mask_folder):
                        mask = cv2.imread(os.path.join(mask_folder, mask_file))
                        if np.all((mask == 0)):
                            rgb_file = os.path.join(image_folder, mask_file)
                            rgb = cv2.imread(rgb_file)
                            if resize:
                                rgb = cv2.resize(rgb, resize_dims)
                            break
                    for mask_file in os.listdir(mask_folder):
                        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
                        if resize:
                            mask = cv2.resize(mask, resize_dims)
                        if not np.all((mask == 0)):
                            rgb_object_file = os.path.join(
                                image_folder, mask_file)
                            rgb_object = cv2.imread(rgb_object_file)
                            if resize:
                                rgb_object = cv2.resize(rgb_object, resize_dims)
                            training_sample = TrainingSample(
                                model_input=ModelInput(rgb=rgb),
                                model_target=ModelTarget(
                                    rgb_with_object=rgb_object,
                                    object_mask=mask
                                )
                            )
                            training_samples.append(training_sample)
        return training_samples

                        









        return training_samples
