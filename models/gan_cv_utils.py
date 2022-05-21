import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple
from data.training_sample import TrainingSample, ModelOutput


class GeneratorConvolutional(nn.Module):
    def __init__(self, noise_dim: int, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()
        # define model
        channels = 1
        self.init_size = img_shape//4
        self.l1 = nn.Sequential(nn.Linear(noise_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )


    def forward(self, z: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.l1(z)
        features = features.view(features.shape[0], 128, self.init_size, self.init_size)

        mask_logits = self.mask_head(features).view(-1, self.img_shape[1], self.img_shape[2])
        masks = torch.sigmoid(mask_logits).unsqueeze(1).detach()
        image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) * masks + rgb * (1.-masks)

        return image, mask_logits

class DiscriminatorConvolutional(nn.Module):
    def __init__(self, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()

        init = RandomNormal(stddev=0.02)
        self.model_rgb = Sequential()
        # downsample to half the size
        self.model_rgb.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=img_shape))
        self.model_rgb.add(LeakyReLU(alpha=0.2))
        # downsample to half the size again
        self.model_rgb.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.model_rgb.add(LeakyReLU(alpha=0.2))
        self.model_rgb.add(Flatten())
        self.model_rgb.add(Dense(1, activation='sigmoid'))
        self.model_rgb.compile()

        self.model_mask = Sequential()
        # downsample to half the size
        self.model_mask.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=img_shape))
        self.model_mask.add(LeakyReLU(alpha=0.2))
        # downsample to half the size again
        self.model_mask.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.model_mask.add(LeakyReLU(alpha=0.2))
        self.model_mask.add(Flatten())
        self.model_mask.add(Dense(1, activation='sigmoid'))
        self.model_mask.compile()
        

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
