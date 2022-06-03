import numpy as np
import re
from tqdm import tqdm
from typing import Tuple, List
import os
import xml.etree.ElementTree as ET
from PIL import Image

import torchvision
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF

from data.training_sample import TrainingSample, ModelInput, ModelTarget
from data.CVAT_reader import read_mask_for_image
from data.data_generator import DataGenerator


class RealDataGenerator(DataGenerator):

    def __init__(self, resize: bool, resize_dims: Tuple[int, int], dataset_dir: str, single_item_box_only: bool) -> None:
        self._resize = resize
        self._resize_dims = resize_dims
        self._dataset_dir = dataset_dir
        self._single_item_box_only = single_item_box_only

    def generate(self) -> List[TrainingSample]:
        """If resize, resize_dims must be provided"""
        cvat_xml = os.path.join(self._dataset_dir, "annotations.xml")
        cvat_xml_root = ET.parse(cvat_xml).getroot()
        image_dir = os.path.join(self._dataset_dir, "images")
        crop_param = (0, 0, 1430, 1080)
        subset_names = [directory for directory in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, directory))]
        samples = []
        if self._single_item_box_only:
            subset_names = [subset_name for subset_name in subset_names if re.match("single_item_box_\d", subset_name)]
        for subset_name in tqdm(subset_names, desc=f"Generating dataset"):
            assert "box" in subset_name, subset_name
            subset_image_dir = os.path.join(image_dir, subset_name)
            previous_rgb_tensor = None
            for next_rgb_filename in tqdm(sorted(os.listdir(subset_image_dir)), desc=f"Processing subset {subset_name}"):
                next_rgb_path = os.path.join(subset_image_dir, next_rgb_filename)
                next_rgb = Image.open(next_rgb_path).crop(crop_param)
                if self._resize:
                    next_rgb = next_rgb.resize(self._resize_dims)
                next_rgb_tensor = TF.to_tensor(next_rgb).float()
                mask = read_mask_for_image(cvat_xml_root=cvat_xml_root, image_filename=next_rgb_filename, subset_name=subset_name)

                if previous_rgb_tensor is not None and np.any(mask):
                    mask_pil = Image.fromarray(mask).crop(crop_param)
                    if self._resize:
                        mask_pil = mask_pil.resize(self._resize_dims, resample=Image.NEAREST)
                    mask_tensor = TF.to_tensor(mask_pil).squeeze(0).bool()
                    bounding_box_xyxy = masks_to_boxes(mask_tensor.unsqueeze(0))[0].int()
                    xmin, ymin, xmax, ymax = bounding_box_xyxy

                    normalized_bounding_box_xyxy = bounding_box_xyxy.clone().float()
                    normalized_bounding_box_xyxy[0] /= mask_tensor.shape[1]
                    normalized_bounding_box_xyxy[1] /= mask_tensor.shape[0]
                    normalized_bounding_box_xyxy[2] /= mask_tensor.shape[1]
                    normalized_bounding_box_xyxy[3] /= mask_tensor.shape[0]
                    zoomed_object_mask = TF.resize(mask_tensor[ymin:ymax + 1, xmin:xmax + 1].unsqueeze(0).float(),
                                                   size=self._resize_dims,
                                                   interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                   ).squeeze(0).bool()
                    zoomed_object_rgb = TF.resize(next_rgb_tensor[:, ymin:ymax + 1, xmin:xmax + 1],
                                                  size=self._resize_dims,
                                                  ).float()

                    training_sample = TrainingSample(
                        model_input=ModelInput(rgb=previous_rgb_tensor.float()),
                        model_target=ModelTarget(
                            rgb_with_object=next_rgb_tensor,
                            object_mask=mask_tensor,
                            normalized_bounding_box_xyxy=normalized_bounding_box_xyxy,
                            zoomed_object_mask=zoomed_object_mask,
                            zoomed_object_rgb=zoomed_object_rgb,
                        ),
                        sample_id=next_rgb_path,
                    )
                    samples.append(training_sample)
                if re.match("single_item_box_\d", subset_name):
                    previous_rgb_tensor = previous_rgb_tensor if previous_rgb_tensor is not None else next_rgb_tensor.clone()
                else:
                    previous_rgb_tensor = next_rgb_tensor.clone()
        return samples
