from abc import ABC
import numpy as np
from typing import Tuple, List
import torch
import os
import cv2
from PIL import Image
import torchvision
from data.training_sample import TrainingSample, ModelInput, ModelTarget


class RealDataGenerator(ABC):
    def __init__(self):
        pass

    def generate(self, num_samples: int, resize: bool, resize_dims: Tuple[int, int]) -> List[TrainingSample]:
        """If resize, resize_dims must be provided"""
        training_samples = []
        masks_dir = "masks"
        image_dir = "images_1"
        crop_param = (0, 0, 1430, 1080)
        # print(os.walk(image_dir))
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for box in dirs:
                if "box" in box:
                    image_folder = os.path.join(root, box)
                    mask_folder = os.path.join(masks_dir, box)
                    # identify empty mask
                    for mask_file in os.listdir(mask_folder):
                        mask = Image.open(os.path.join(mask_folder, mask_file)).crop(crop_param)
                        mask_tensor = torchvision.transforms.ToTensor()(mask)
                        if torch.all((mask_tensor == 0)):
                            rgb_file = os.path.join(image_folder, mask_file)
                            rgb = Image.open(rgb_file).crop(crop_param)
                            if resize:
                                rgb = rgb.resize(resize_dims)
                            rgb_tensor = torchvision.transforms.ToTensor()(rgb)
                            break
                    for mask_file in os.listdir(mask_folder):
                        mask = Image.open(os.path.join(mask_folder, mask_file)).convert('L').crop(crop_param)
                        if resize:
                            mask = mask.resize(resize_dims, resample=Image.NEAREST)
                        mask_tensor = torchvision.transforms.ToTensor()(mask)
                        if not torch.all((mask_tensor == 0)):
                            rgb_object_file = os.path.join(
                                image_folder, mask_file)
                            rgb_object = Image.open(rgb_object_file).crop(crop_param)
                            if resize:
                                rgb_object = rgb_object.resize(resize_dims)
                            rgb_object_tensor = torchvision.transforms.ToTensor()(rgb_object)
                            training_sample = TrainingSample(
                                model_input=ModelInput(rgb=rgb_tensor.float()),
                                model_target=ModelTarget(
                                    rgb_with_object=rgb_object_tensor.float(),
                                    object_mask=mask_tensor.squeeze(0).bool()
                                )
                            )
                            training_samples.append(training_sample)
        return training_samples