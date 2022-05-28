import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple
from data.training_sample import TrainingSample, ModelOutput


class GeneratorFullyConnected(nn.Module):
    def __init__(self, noise_dim: int, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()
        self.img_shape = img_shape
        common_layers = []
        input_dim = noise_dim+np.prod(img_shape)
        for hdim in hidden_dims:
            common_layers.append(nn.Linear(input_dim, hdim))
            common_layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hdim
        self.feature_extractor = nn.Sequential(*self.decoder_layers)

        self.feature_extractor = nn.Sequential(*common_layers)
        self.image_head = nn.Linear(input_dim, np.prod(img_shape))
        self.mask_head = nn.Linear(input_dim, np.prod(img_shape[1:]))

    def forward(self, z: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_flat = torch.flatten(rgb, start_dim=1)
        features = self.feature_extractor(torch.concat([z, rgb_flat], dim=1))
        mask_logits = self.mask_head(features).view(-1, self.img_shape[1], self.img_shape[2])
        masks = torch.sigmoid(mask_logits).unsqueeze(1).detach()
        image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) * masks + rgb * (1.-masks)

        return image, mask_logits

class DiscriminatorFullyConnected(nn.Module):
    def __init__(self, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()
        rgb_input_dim = np.prod(img_shape)
        rgb_layers = [nn.Flatten()]
        for i, h in enumerate(hidden_dims):
            rgb_layers.append(nn.Linear(hidden_dims[i-1] if i > 0 else rgb_input_dim, h))
            rgb_layers.append(nn.ReLU())
        rgb_layers.append(nn.Linear(hidden_dims[-1], 1))
        rgb_layers.append(nn.Sigmoid())
        mask_input_dim = np.prod(img_shape[1:])
        mask_layers = [nn.Flatten()]
        for i, h in enumerate(hidden_dims):
            mask_layers.append(nn.Linear(hidden_dims[i-1] if i > 0 else mask_input_dim, h))
            mask_layers.append(nn.ReLU())
        mask_layers.append(nn.Linear(hidden_dims[-1], 1))
        mask_layers.append(nn.Sigmoid())

        self.model_rgb = nn.Sequential(*rgb_layers)
        self.model_mask = nn.Sequential(*mask_layers)

    def forward(self, batch: TrainingSample)->Tuple[float, float]:
        try:
            rgb = batch['model_target']['rgb_with_object']
            mask = batch['model_target']['object_mask']
        except:
            rgb = batch['rgb_with_object']
            mask = batch['soft_object_mask']
        rgb_flat = rgb.view(rgb.size(0), -1)
        mask_flat = mask.view(mask.size(0), -1)
        validity_rgb = self.model_rgb(rgb_flat)
        validity_mask = self.model_mask(mask_flat.float())
        return (validity_rgb, validity_mask)
