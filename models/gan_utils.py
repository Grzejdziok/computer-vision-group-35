from sys import float_repr_style
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from typing import List, Tuple
from data.training_sample import TrainingSample, ModelOutput


class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, img_shape: Tuple[int, int, int]):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat:int, out_feat:int, normalize:bool=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        input_dim = latent_dim+np.prod(img_shape)
        self.feature_extractor = nn.Sequential(
            *block(input_dim, hidden_dim, normalize=False),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim)
        )

        self.image_head = nn.Linear(hidden_dim, int(np.prod(img_shape)))
        self.mask_head = nn.Linear(hidden_dim, int(np.prod(img_shape[1:])))

    def forward(self, z: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_flat = torch.flatten(rgb, start_dim=1)
        features = self.feature_extractor(torch.concat([z, rgb_flat], dim=1))
        mask_logits = self.mask_head(features).view(-1, self.img_shape[1], self.img_shape[2])
        masks = torch.sigmoid(mask_logits).unsqueeze(1).detach()
        image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) * masks + rgb * (1.-masks)

        return image, mask_logits

class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int]):
        super().__init__()

        self.model_rgb = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.model_mask = nn.Sequential(
            nn.Linear(int(np.prod(img_shape[1:])), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

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
