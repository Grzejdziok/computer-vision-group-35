from abc import ABC
from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms
import torchvision.transforms.functional as TF


class EncoderLocal(ABC, nn.Module):

    def forward(self, normalized_bounding_box_xyxy: torch.Tensor, zoomed_object_mask: torch.Tensor, zoomed_object_rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def kl(self) -> torch.Tensor:
        raise NotImplementedError()


class DecoderLocal(ABC, nn.Module):

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class EncoderLocalFullyConnected(EncoderLocal):

    def __init__(self, latent_dims: int, input_image_size: int, hdim: List[int]):
        super(EncoderLocalFullyConnected, self).__init__()

        self.input_image_size = input_image_size
        input_dim = input_image_size**2*4+4
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

    def forward(self, normalized_bounding_box_xyxy: torch.Tensor, zoomed_object_mask: torch.Tensor, zoomed_object_rgb: torch.Tensor) -> torch.Tensor:
        rgb_with_object_resized = TF.resize(
            img=zoomed_object_rgb,
            size=(self.input_image_size, self.input_image_size),
        )
        object_mask_resized = TF.resize(
            img=zoomed_object_mask.float(),
            size=(self.input_image_size, self.input_image_size),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        rgb_with_object_flat = torch.flatten(rgb_with_object_resized, start_dim=1)
        object_mask_flat = torch.flatten(object_mask_resized, start_dim=1)
        encoder_input = torch.concat([rgb_with_object_flat, object_mask_flat, normalized_bounding_box_xyxy], dim=1)
        features = self.feature_extractor(encoder_input)
        sig = torch.exp(self.sigma_head(features))  # make it stay positive
        mu = self.mean_head(features)
        #reparameterize to find z
        z = self.reparameterize(mu, sig)
        #loss between N(0,I) and learned distribution
        self._kl = self.kull_leib(mu, sig)
        return z


class DecoderLocalFullyConnected(DecoderLocal):

    def __init__(self, latent_dims: int, model_output_image_size: int, output_image_size: int, hdim: List[int]):
        super(DecoderLocalFullyConnected, self).__init__()

        rgb_embedding_dims = model_output_image_size
        common_layers = []
        in_features = latent_dims + rgb_embedding_dims
        for h in hdim[::-1]:
            common_layers.append(nn.Linear(in_features, h))
            common_layers.append(nn.ReLU())
            in_features = h

        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()
        resnet18.requires_grad_(False)
        self.rgb_embedding = nn.Sequential(resnet18, nn.Linear(512, rgb_embedding_dims))
        self.feature_extractor = nn.Sequential(*common_layers)
        self.image_head = nn.Linear(in_features, model_output_image_size**2*3)
        self.mask_head = nn.Linear(in_features, model_output_image_size**2)
        self.box_head = nn.Linear(in_features, 4)
        self.model_output_image_size = model_output_image_size
        self.output_image_size = output_image_size

    def forward(self, z: torch.Tensor, normalized_rgb: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_embedded = self.rgb_embedding(normalized_rgb)
        features = self.feature_extractor(torch.concat([z, rgb_embedded], dim=1))
        mask_logits = self.mask_head(features).view(-1, self.model_output_image_size, self.model_output_image_size)
        image = self.image_head(features).view(-1, 3, self.model_output_image_size, self.model_output_image_size)
        boxes_xywh = torch.sigmoid(self.box_head(features))
        boxes = torch.concat([boxes_xywh[:, :2], boxes_xywh[:, :2] + boxes_xywh[:, 2:]], dim=1)

        mask_logits = TF.resize(mask_logits, (self.output_image_size, self.output_image_size))
        image = TF.resize(image, (self.output_image_size, self.output_image_size))

        return image, mask_logits, boxes

"""
class EncoderLocalConvolutional(EncoderLocal):

    def __init__(self, latent_dims: int, input_image_size: int, hidden_channels: List[int]):
        super(EncoderLocalConvolutional, self).__init__()

        self.input_image_size = input_image_size
        in_channels = 3 + 1 + 4  # rgb + mask + box
        common_layers = [nn.Flatten()]
        for out_channels in hidden_channels:
            common_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            common_layers.append(nn.ReLU())
            in_channels = out_channels

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

    def forward(self, normalized_bounding_box_xyxy: torch.Tensor, zoomed_object_mask: torch.Tensor, zoomed_object_rgb: torch.Tensor) -> torch.Tensor:
        rgb_with_object_resized = TF.resize(
            img=zoomed_object_rgb,
            size=(self.input_image_size, self.input_image_size),
        )
        object_mask_resized = TF.resize(
            img=zoomed_object_mask.float(),
            size=(self.input_image_size, self.input_image_size),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        rgb_with_object_flat = torch.flatten(rgb_with_object_resized, start_dim=1)
        object_mask_flat = torch.flatten(object_mask_resized, start_dim=1)
        encoder_input = torch.concat([rgb_with_object_flat, object_mask_flat, normalized_bounding_box_xyxy], dim=1)
        features = self.feature_extractor(encoder_input)
        sig = torch.exp(self.sigma_head(features))  # make it stay positive
        mu = self.mean_head(features)
        #reparameterize to find z
        z = self.reparameterize(mu, sig)
        #loss between N(0,I) and learned distribution
        self._kl = self.kull_leib(mu, sig)
        return z
"""