import numpy as np
from abc import ABC
import re
from tqdm import tqdm
from typing import Tuple, List
import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms.functional as TF
from data.training_sample import TrainingSample, ModelInput, ModelTarget
from data.CVAT_reader import read_mask_for_image
import json


class RealDataGenerator:

    def generate(self, resize: bool, resize_dims: Tuple[int, int], dataset_dir: str, datamodule_dir: str) -> List[TrainingSample]:
        """If resize, resize_dims must be provided"""
        if not os.path.exists(datamodule_dir):
            cvat_xml = os.path.join(dataset_dir, "annotations.xml")
            cvat_xml_root = ET.parse(cvat_xml).getroot()
            image_dir = os.path.join(dataset_dir, "images")
            samples = []
            crop_param = (0, 0, 1430, 1080)
            subset_names = [directory for directory in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, directory))]
            for subset_name in tqdm(subset_names, desc=f"Generating dataset"):
                assert "box" in subset_name, subset_name
                subset_image_dir = os.path.join(image_dir, subset_name)
                previous_rgb_tensor = None
                for next_rgb_filename in tqdm(sorted(os.listdir(subset_image_dir)), desc=f"Processing subset {subset_name}"):
                    next_rgb_path = os.path.join(subset_image_dir, next_rgb_filename)
                    next_rgb = Image.open(next_rgb_path).crop(crop_param)
                    if resize:
                        next_rgb = next_rgb.resize(resize_dims)
                    next_rgb_tensor = TF.to_tensor(next_rgb)
                    mask = read_mask_for_image(cvat_xml_root=cvat_xml_root, image_filename=next_rgb_filename, subset_name=subset_name)

                    if previous_rgb_tensor is not None and np.any(mask):
                        mask_pil = Image.fromarray(mask).crop(crop_param)
                        if resize:
                            mask_pil = mask_pil.resize(resize_dims, resample=Image.NEAREST)
                        mask_tensor = TF.to_tensor(mask_pil)
                        training_sample = TrainingSample(
                            model_input=ModelInput(rgb=previous_rgb_tensor.float()),
                            model_target=ModelTarget(
                                rgb_with_object=next_rgb_tensor.float(),
                                object_mask=mask_tensor.squeeze(0).bool()
                            )
                        )
                        samples.append(training_sample)
                    if re.match("single_item_box_\d", subset_name):
                        previous_rgb_tensor = previous_rgb_tensor if previous_rgb_tensor is not None else next_rgb_tensor.clone()
                    else:
                        previous_rgb_tensor = next_rgb_tensor.clone()

            #I was in a rush I will make it pretty later
            for image in samples:
                image['model_input']['rgb'] = image['model_input']['rgb'].tolist()
                image['model_target']['rgb_with_object'] = image['model_target']['rgb_with_object'].tolist()
                image['model_target']['object_mask'] = image['model_target']['object_mask'].tolist()
            dm_dict = {'samples': samples}
            with open(datamodule_dir, "w") as dm_json:
                json.dump(dm_dict, dm_json)

        with open(datamodule_dir) as f:
            dm_dict = json.load(f)

        for image in dm_dict['samples']:
            image['model_input']['rgb'] = torch.Tensor(image['model_input']['rgb'])
            image['model_target']['rgb_with_object'] = torch.Tensor(image['model_target']['rgb_with_object'])
            image['model_target']['object_mask'] = torch.Tensor(image['model_target']['object_mask']).bool()

        return dm_dict['samples']
