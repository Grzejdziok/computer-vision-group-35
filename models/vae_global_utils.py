from abc import ABC
from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class EncoderGlobal(ABC, nn.Module):

    def forward(self, rgb_with_object: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def kl(self) -> torch.Tensor:
        raise NotImplementedError()


class DecoderGlobal(ABC, nn.Module):

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class EncoderGlobalFullyConnected(EncoderGlobal):

    def __init__(self, latent_dims: int, input_image_size: int, hdim: List[int]):
        super(EncoderGlobalFullyConnected, self).__init__()

        self.input_image_size = input_image_size
        input_dim = input_image_size**2*4
        common_layers = [nn.Flatten()]
        for h in hdim:
            common_layers.append(nn.Linear(input_dim, h))
            common_layers.append(nn.ReLU())
            input_dim = h

        self.feature_extractor = nn.Sequential(*common_layers)
        self.mean_head = nn.Linear(input_dim, latent_dims)
        self.sigma_head = nn.Linear(input_dim, latent_dims)

        #distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self._kl = torch.zeros((1, ))

    @property
    def kl(self) -> torch.Tensor:
        return self._kl

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        return self.N.sample((num_samples, self.mean_head.out_features)).to(device)

    def reparameterize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return mu + sig * self.sample(num_samples=mu.shape[0], device=mu.device)

    def kull_leib(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(torch.distributions.Normal(mu, sig), self.N).mean()

    def forward(self, rgb_with_object: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        rgb_with_object_resized = TF.resize(rgb_with_object, (self.input_image_size, self.input_image_size))
        object_mask_resized = TF.resize(object_mask, (self.input_image_size, self.input_image_size))

        rgb_with_object_flat = torch.flatten(rgb_with_object_resized, start_dim=1)
        object_mask_flat = torch.flatten(object_mask_resized.float(), start_dim=1)
        encoder_input = torch.concat([rgb_with_object_flat, object_mask_flat], dim=1)
        features = self.feature_extractor(encoder_input)
        sig = torch.exp(self.sigma_head(features))  # make it stay positive
        mu = self.mean_head(features)
        #reparameterize to find z
        z = self.reparameterize(mu, sig)
        #loss between N(0,I) and learned distribution
        self._kl = self.kull_leib(mu, sig)
        return z


class DecoderGlobalFullyConnected(DecoderGlobal):
    def __init__(self, latent_dims: int, model_output_image_size: int, output_image_size: int, hdim: List[int]):
        super(DecoderGlobalFullyConnected, self).__init__()

        rgb_embedding_dims = model_output_image_size
        common_layers = []
        in_features = latent_dims + rgb_embedding_dims
        for h in hdim[::-1]:
            common_layers.append(nn.Linear(in_features, h))
            common_layers.append(nn.ReLU())
            in_features = h

        self.rgb_embedding = nn.Sequential(
            nn.Linear(output_image_size**2*3, rgb_embedding_dims**2),
            nn.ReLU(),
            nn.Linear(rgb_embedding_dims**2, rgb_embedding_dims),
        )
        self.feature_extractor = nn.Sequential(*common_layers)
        self.image_head = nn.Linear(in_features, model_output_image_size**2*3)
        self.mask_head = nn.Linear(in_features, model_output_image_size**2)
        self.model_output_image_size = model_output_image_size
        self.output_image_size = output_image_size

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_rgb_flat = torch.flatten(normalized_rgb, start_dim=1)
        rgb_embedded = self.rgb_embedding(normalized_rgb_flat)
        features = self.feature_extractor(torch.concat([z, rgb_embedded], dim=1))
        mask_logits = self.mask_head(features).view(-1, self.model_output_image_size, self.model_output_image_size)
        masks = (torch.sigmoid(mask_logits).unsqueeze(1).detach() > 0.5).float()
        image = self.image_head(features).view(-1, 3, self.model_output_image_size, self.model_output_image_size) * masks \
                + rgb * (1.-masks)

        mask_logits = TF.resize(mask_logits, (self.output_image_size, self.output_image_size))
        image = TF.resize(image, (self.output_image_size, self.output_image_size))

        return image, mask_logits
