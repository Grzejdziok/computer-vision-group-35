from typing import Tuple, List
import uuid

import torch
import torchvision.transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF

from data.data_generator import DataGenerator
from data.training_sample import TrainingSample, ModelInput, ModelTarget


class GaussianNoiseWithSquareSyntheticDataGenerator(DataGenerator):

    def __init__(self, image_size: Tuple[int, int], square_size: int, num_samples: int):
        self.image_size = image_size
        self.square_size = square_size
        self._num_samples = num_samples

    def generate(self) -> List[TrainingSample]:
        training_samples = []
        for i in range(self._num_samples):
            rgb = torch.normal(0., 1., (3, self.image_size[0], self.image_size[1]))
            top = torch.randint(0, self.image_size[0] - self.square_size, (1,))
            left = torch.randint(0, self.image_size[1] - self.square_size, (1,))
            object_mask = torch.zeros((self.image_size[0], self.image_size[1])).bool()
            object_mask[top:top + self.square_size, left:left + self.square_size] = True
            object_color = torch.rand((3,))
            rgb_with_object = rgb.clone()
            rgb_with_object[:, object_mask] = object_color

            bounding_box_xyxy = masks_to_boxes(object_mask.unsqueeze(0))[0].int()
            xmin, ymin, xmax, ymax = bounding_box_xyxy

            normalized_bounding_box_xyxy = bounding_box_xyxy.clone().float()
            normalized_bounding_box_xyxy[0] /= object_mask.shape[1]
            normalized_bounding_box_xyxy[1] /= object_mask.shape[0]
            normalized_bounding_box_xyxy[2] /= object_mask.shape[1]
            normalized_bounding_box_xyxy[3] /= object_mask.shape[0]

            zoomed_object_mask = TF.resize(object_mask[ymin:ymax+1, xmin:xmax+1].unsqueeze(0).float(),
                                           size=(self.image_size, self.image_size),
                                           interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                           ).squeeze(0).bool()
            zoomed_object_rgb = TF.resize(rgb_with_object[:, ymin:ymax+1, xmin:xmax+1],
                                          size=(self.image_size, self.image_size),
                                          ).float()

            training_sample = TrainingSample(
                model_input=ModelInput(rgb=rgb.float()),
                model_target=ModelTarget(
                    rgb_with_object=rgb_with_object,
                    object_mask=object_mask,
                    normalized_bounding_box_xyxy=normalized_bounding_box_xyxy,
                    zoomed_object_mask=zoomed_object_mask,
                    zoomed_object_rgb=zoomed_object_rgb,
                ),
                sample_id=str(uuid.uuid4()),
            )
            training_samples.append(training_sample)

        return training_samples
