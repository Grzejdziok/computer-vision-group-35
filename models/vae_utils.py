from typing import List, Tuple

import torch
import torch.nn as nn


class EncoderFullyConnected(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
        super(EncoderFullyConnected, self).__init__()

        input_dim = s_img**2*3
        common_layers = [nn.Flatten()]
        for i, h in enumerate(hdim):
            common_layers.append(nn.Linear(hdim[i-1] if i > 0 else input_dim, h))
            common_layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*common_layers)
        self.mean_head = nn.Linear(hdim[-1], latent_dims)
        self.sigma_head = nn.Linear(hdim[-1], latent_dims)

        #distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def kull_leib(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(torch.distributions.Normal(mu, sig), self.N)[:, 0].sum()

    def reparameterize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return mu + sig*self.N.sample(mu.shape).to(mu.device)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(rgb)
        sig = torch.exp(self.sigma_head(features))  # make it stay positive
        mu = self.mean_head(features)
        #reparameterize to find z
        z = self.reparameterize(mu, sig)
        #loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)
        return z


class DecoderFullyConnected(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
        super(DecoderFullyConnected, self).__init__()

        common_layers = []
        for i, h in enumerate(hdim[::-1]):
            common_layers.append(nn.Linear(hdim[-i+1] if i > 0 else latent_dims, h))
            common_layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*common_layers)
        self.image_head = nn.Linear(hdim[0], s_img**2*3)
        self.mask_head = nn.Linear(hdim[0], s_img**2)
        self.s_img = s_img

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(z)
        image = self.image_head(features).view(-1, self.s_img, self.s_img, 3)
        mask = self.mask_head(features).view(-1, self.s_img, self.s_img)
        return image, mask