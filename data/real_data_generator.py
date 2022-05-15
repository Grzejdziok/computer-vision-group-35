from abc import ABC
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, List
import torch
import os
from PIL import Image
import torchvision
from data.training_sample import TrainingSample, ModelInput, ModelTarget
from random import sample
import json


class RealDataGenerator(ABC):
    def __init__(self):
        pass

    def _total_samples(self, image_dir: str):
        return sum([len(files) for _, _, files in os.walk(image_dir)])-sum([len(dirs) for _, dirs, _ in os.walk(image_dir)])

    def generate(self, dataset: str, train_ratio: int, resize: bool, resize_dims: Tuple[int, int], masks_dir: str, image_dir: str, datamodule_dir: str) -> List[TrainingSample]:
        """If resize, resize_dims must be provided"""
        if not os.path.exists(datamodule_dir):
            samples = []
            crop_param = (0, 0, 1430, 1080)
            boxes = os.listdir(image_dir)
            for box in tqdm(boxes, desc=f"Generating dataset"):
                if "box" in box:
                    image_folder = os.path.join(image_dir, box)
                    mask_folder = os.path.join(masks_dir, box)
                    # identify empty mask
                    for mask_file in os.listdir(mask_folder):
                        mask = Image.open(os.path.join(
                            mask_folder, mask_file)).crop(crop_param)
                        mask_tensor = torchvision.transforms.ToTensor()(mask)
                        if torch.all((mask_tensor == 0)):
                            rgb_file = os.path.join(image_folder, mask_file)
                            rgb = Image.open(rgb_file).crop(crop_param)
                            if resize:
                                rgb = rgb.resize(resize_dims)
                            rgb_tensor = torchvision.transforms.ToTensor()(rgb)
                            break
                    for mask_file in os.listdir(mask_folder):
                        mask = Image.open(os.path.join(mask_folder, mask_file)).convert(
                            'L').crop(crop_param)
                        if resize:
                            mask = mask.resize(
                                resize_dims, resample=Image.NEAREST)
                        mask_tensor = torchvision.transforms.ToTensor()(mask)
                        if not torch.all((mask_tensor == 0)):
                            rgb_object_file = os.path.join(
                                image_folder, mask_file)
                            rgb_object = Image.open(
                                rgb_object_file).crop(crop_param)
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
                            samples.append(training_sample)

            #I was in a rush I will make it pretty later
            for image in samples:
                image['model_input']['rgb'] = image['model_input']['rgb'].tolist()
                image['model_target']['rgb_with_object'] = image['model_target']['rgb_with_object'].tolist()
                image['model_target']['object_mask'] = image['model_target']['object_mask'].tolist()
            samples_train, samples_test = train_test_split(samples, train_size=train_ratio, random_state=2022)
            dm_dict = {'samples_train': samples_train, 'samples_test':samples_test}
            with open(datamodule_dir, "w") as dm_json:
                json.dump(dm_dict, dm_json)

        f = open(datamodule_dir)
        dm_dict = json.load(f)
        for image in dm_dict['samples_train']:
            image['model_input']['rgb'] = torch.Tensor(image['model_input']['rgb'])
            image['model_target']['rgb_with_object'] = torch.Tensor(image['model_target']['rgb_with_object'])
            image['model_target']['object_mask'] = torch.Tensor(image['model_target']['object_mask']).bool()
        for image in dm_dict['samples_test']:
            image['model_input']['rgb'] = torch.Tensor(image['model_input']['rgb'])
            image['model_target']['rgb_with_object'] = torch.Tensor(image['model_target']['rgb_with_object'])
            image['model_target']['object_mask'] = torch.Tensor(image['model_target']['object_mask']).bool()

        if dataset == "train":
            return dm_dict['samples_train']
        elif dataset == "test":
            return dm_dict['samples_test']
        
