import numpy as np
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple
from data.training_sample import TrainingSample, ModelOutput
import torchvision.transforms.functional as TF
import torchvision.transforms

class GeneratorLocal(ABC, nn.Module):

    def forward(self, normalized_bounding_box_xyxy: torch.Tensor, zoomed_object_mask: torch.Tensor, zoomed_object_rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class DiscriminatorLocal(ABC, nn.Module):

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class GeneratorFullyConnected(GeneratorLocal):
    def __init__(self, noise_dim: int, hidden_dims: List[int], img_shape: Tuple[int, int], model_output_image_size: int):
        super().__init__()
        
        self.img_shape = img_shape
        rgb_embedding_dims = model_output_image_size
        in_features = noise_dim + rgb_embedding_dims
        common_layers = []
        for h in hidden_dims[::-1]:
            common_layers.append(nn.Linear(in_features, h))
            common_layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = h
        
        self.rgb_embedding = nn.Sequential(
            nn.Linear(np.prod(img_shape)*3, rgb_embedding_dims**2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(rgb_embedding_dims**2, rgb_embedding_dims),
        )

        self.feature_extractor = nn.Sequential(*common_layers)
        self.image_head = nn.Linear(in_features, 3*np.prod(img_shape))
        self.mask_head = nn.Linear(in_features, np.prod(img_shape))
        self.box_head = nn.Linear(in_features, 4)
        self.model_output_image_size = model_output_image_size

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # rgb_flat = torch.flatten(rgb, start_dim=1)
        # features = self.feature_extractor(torch.concat([z, rgb_flat], dim=1))
        # mask_logits = self.mask_head(features).view(-1, self.img_shape[1], self.img_shape[2])
        # masks = torch.sigmoid(mask_logits).unsqueeze(1).detach()
        # image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) * masks + rgb * (1.-masks)

        # return image, mask_logits

        normalized_rgb_flat = torch.flatten(normalized_rgb, start_dim=1)
        rgb_embedded = self.rgb_embedding(normalized_rgb_flat)
        features = self.feature_extractor(torch.concat([z, rgb_embedded], dim=1))
        # mask_logits = self.mask_head(features).view(-1, self.model_output_image_size, self.model_output_image_size)
        mask_logits = torch.sigmoid(self.mask_head(features).view(-1, self.model_output_image_size, self.model_output_image_size))
        # image = self.image_head(features).view(-1, 3, self.model_output_image_size, self.model_output_image_size)
        image = torch.sigmoid(self.image_head(features).view(-1, 3, self.model_output_image_size, self.model_output_image_size))
        boxes_xywh = torch.sigmoid(self.box_head(features))
        boxes = torch.concat([boxes_xywh[:, :2], boxes_xywh[:, :2] + boxes_xywh[:, 2:]], dim=1)
        mask_logits = TF.resize(mask_logits, (self.img_shape[0], self.img_shape[1]))
        image = TF.resize(image, (self.img_shape[0], self.img_shape[1]))
        return image, mask_logits, boxes


class DiscriminatorFullyConnected(DiscriminatorLocal):
    def __init__(self, hidden_dims: List[int], img_shape: Tuple[int, int]):
        super().__init__()
        self.img_shape = img_shape

        #rgb+mask
        input_dim = 3*np.prod(img_shape) #rgb
        common_layers = [nn.Flatten()]
        common_layers = []
        for h in hidden_dims:
            common_layers.append(nn.Linear(input_dim, h))
            common_layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = h
        common_layers.append(nn.Linear(hidden_dims[-1], 1))
        common_layers.append(nn.Sigmoid())
        self.feature_extractor = nn.Sequential(*common_layers)

        #box
        box_layers = [nn.Linear(4, 2), nn.LeakyReLU(0.2, inplace=True), nn.Linear(2, 1), nn.Sigmoid()]
        self.box_model = nn.Sequential(*box_layers)

    def forward(self, batch: TrainingSample) -> torch.Tensor:
        rgb_with_object = batch['rgb_with_object']
        object_mask = batch['soft_object_mask']
        rgb_with_object_resized = TF.resize(
            img=rgb_with_object,
            size=(self.img_shape[0], self.img_shape[1]),
        )
        object_mask_resized = TF.resize(
            img=object_mask.float(),
            size=(self.img_shape[0], self.img_shape[1]),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        rgb_with_object_flat = torch.flatten(rgb_with_object_resized, start_dim=1)
        validity = self.feature_extractor(rgb_with_object_flat)
        return validity