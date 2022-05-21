from cv2 import mean
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple
from data.training_sample import TrainingSample, ModelOutput
from PIL import Image


class GeneratorFullyConnected(nn.Module):
    def __init__(self, noise_dim: int, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()
        # self.img_shape = img_shape
        # common_layers = []
        # in_features = noise_dim+np.prod(img_shape)
        # for h in hidden_dims[::-1]:
        #     common_layers.append(nn.Linear(in_features, h))
        #     common_layers.append(nn.ReLU())
        #     in_features = h

        # self.feature_extractor = nn.Sequential(*common_layers)
        # self.image_head = nn.Linear(in_features, np.prod(img_shape))
        # self.mask_head = nn.Linear(in_features, np.prod(img_shape[1:]))

        #From original GAN paper
        # in_features = noise_dim+np.prod(img_shape)
        self.img_shape = img_shape
        in_features = noise_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.feature_layers = []
        for i, hdim in enumerate(reversed(hidden_dims)):
            if i==0:
                self.feature_layers+=block(in_features, hdim, normalize=False)
            else:
                self.feature_layers+=block(in_features, hdim)
            in_features = hdim
        self.feature_extractor = nn.Sequential(*self.feature_layers)

        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dims[0], int(np.prod(img_shape[1:])))
        )
        self.image_head = nn.Sequential(
            nn.Linear(hidden_dims[0], int(np.prod(img_shape)))
        )

    def forward(self, z: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_flat = torch.flatten(rgb, start_dim=1)
        # features = self.feature_extractor(torch.concat([z, rgb_flat], dim=1))
        features = self.feature_extractor(z)
        mask_logits = self.mask_head(features).view(-1, self.img_shape[1], self.img_shape[2])
        masks = (torch.sigmoid(mask_logits).unsqueeze(1).detach() > 0.5).float()
        image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) * masks + rgb * (1.-masks)
        # image = self.image_head(features).view(-1, 3, self.img_shape[1], self.img_shape[2]) + rgb

        return image, mask_logits

class DiscriminatorFullyConnected(nn.Module):
    def __init__(self, hidden_dims: List[int], img_shape: Tuple[int, int, int]):
        super().__init__()
        # rgb_input_dim = np.prod(img_shape)
        # rgb_layers = [nn.Flatten()]
        # for i, h in enumerate(hidden_dims):
        #     rgb_layers.append(nn.Linear(hidden_dims[i-1] if i > 0 else rgb_input_dim, h))
        #     rgb_layers.append(nn.ReLU())
        # rgb_layers.append(nn.Linear(hidden_dims[-1], 1))
        # rgb_layers.append(nn.Sigmoid())
        # mask_input_dim = np.prod(img_shape[1:])
        # mask_layers = [nn.Flatten()]
        # for i, h in enumerate(hidden_dims):
        #     mask_layers.append(nn.Linear(hidden_dims[i-1] if i > 0 else mask_input_dim, h))
        #     mask_layers.append(nn.ReLU())
        # mask_layers.append(nn.Linear(hidden_dims[-1], 1))
        # mask_layers.append(nn.Sigmoid())

        # self.model_rgb = nn.Sequential(*rgb_layers)
        # self.model_mask = nn.Sequential(*mask_layers)



        self.decoder_layers = []
        input_dim = np.prod(img_shape)+np.prod(img_shape[1:])
        for hdim in hidden_dims:
            self.decoder_layers.append(nn.Linear(input_dim, hdim))
            self.decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hdim
        self.feature_extractor = nn.Sequential(*self.decoder_layers)

        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        self.image_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, batch: TrainingSample)->Tuple[float, float]:
        if 'model_target' in batch.keys(): #real sample
            rgb = batch['model_target']['rgb_with_object']
            mask = batch['model_target']['object_mask']
            sample = 'real'
        else:   #fake sample
            rgb = batch['rgb_with_object']
            mask = batch['soft_object_mask']
            sample = 'fake'
        
        rgb_flat = rgb.view(rgb.size(0), -1)
        mask_flat = mask.view(mask.size(0), -1)
        discriminator_input = torch.concat([rgb_flat, mask_flat], dim=1)
        features = self.feature_extractor(discriminator_input)
        validity_rgb = self.image_head(features)
        validity_mask = self.mask_head(features)
        return (validity_rgb, validity_mask)
