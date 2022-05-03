from abc import ABC
from turtle import color
from typing import Tuple, List
import numpy as np

import torch

from data.training_sample import TrainingSample, ModelInput, ModelTarget


class SyntheticDataGenerator(ABC):

    def __init__(self, image_size: Tuple[int, int], square_size: int):
        self.image_size = image_size
        self.square_size = square_size

    def generate(self, num_samples: int) -> List[TrainingSample]:
        training_samples = []
        for i in range(num_samples):
            rgb = torch.from_numpy(np.full((self.image_size[0], self.image_size[1], 3), torch.rand((1,3))))
            top = torch.randint(0, self.image_size[0] - self.square_size, (1,))
            left = torch.randint(0, self.image_size[1] - self.square_size, (1,))
            object_mask = torch.zeros((self.image_size[0], self.image_size[1])).bool()
            object_mask[top:top + self.square_size, left:left + self.square_size] = True
            object_color = torch.rand((3,))
            rgb_with_object = rgb.clone()
            rgb_with_object[object_mask] = object_color

            training_sample = TrainingSample(
                model_input=ModelInput(rgb=rgb.float()),
                model_target=ModelTarget(
                    rgb_with_object=rgb_with_object,
                    object_mask=object_mask,
                )
            )
            training_samples.append(training_sample)

        return training_samples


class GaussianNoiseWithSquareSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self, image_size: Tuple[int, int], square_size: int):
        self.image_size = image_size
        self.square_size = square_size

    def generate(self, num_samples: int) -> List[TrainingSample]:
        training_samples = []
        for i in range(num_samples):
            rgb = torch.normal(0., 1., (self.image_size[0], self.image_size[1], 3))
            top = torch.randint(0, self.image_size[0] - self.square_size, (1,))
            left = torch.randint(0, self.image_size[1] - self.square_size, (1,))
            object_mask = torch.zeros((self.image_size[0], self.image_size[1])).bool()
            object_mask[top:top + self.square_size, left:left + self.square_size] = True
            object_color = torch.rand((3,))
            rgb_with_object = rgb.clone()
            rgb_with_object[object_mask] = object_color

            training_sample = TrainingSample(
                model_input=ModelInput(rgb=rgb.float()),
                model_target=ModelTarget(
                    rgb_with_object=rgb_with_object,
                    object_mask=object_mask,
                )
            )
            training_samples.append(training_sample)

        return training_samples
