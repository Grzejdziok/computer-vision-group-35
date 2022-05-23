from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF


<<<<<<< HEAD
class EncoderFullyConnected(nn.Module):
=======
class Encoder(ABC, nn.Module):

    def forward(self, object_rgb: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def kl(self) -> torch.Tensor:
        raise NotImplementedError()


class Decoder(ABC, nn.Module):

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


def _create_fully_connected_layers(input_dim: int, hdim: List[int]) -> Tuple[nn.Sequential, int]:
    common_layers = [nn.Flatten()]
    for h in hdim:
        common_layers.append(nn.Linear(input_dim, h))
        common_layers.append(nn.ReLU())
        input_dim = h

    return nn.Sequential(*common_layers), input_dim


class EncoderFullyConnected(Encoder):
>>>>>>> d726904 (WIP)

    def __init__(self, latent_dims: int, model_image_size: int, hdim: List[int]):
        super(EncoderFullyConnected, self).__init__()

<<<<<<< HEAD
        input_dim = s_img**2*4
        common_layers = [nn.Flatten()]
        for i, h in enumerate(hdim):
            common_layers.append(nn.Linear(hdim[i-1] if i > 0 else input_dim, h))
            common_layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*common_layers)
        self.mean_head = nn.Linear(hdim[-1], latent_dims)
        self.sigma_head = nn.Linear(hdim[-1], latent_dims)
=======
        self.model_image_size = model_image_size
        input_dim = self.model_image_size**2
        self.feature_extractor_mask, out_features = _create_fully_connected_layers(input_dim=input_dim, hdim=hdim)
        self.feature_extractor_object, out_features = _create_fully_connected_layers(input_dim=input_dim*3, hdim=hdim)
        self.feature_combiner = nn.Sequential(nn.Linear(out_features*2, out_features), nn.ReLU())
        self.mean_head = nn.Linear(out_features, latent_dims)
        self.sigma_head = nn.Linear(out_features, latent_dims)
>>>>>>> d726904 (WIP)

        #distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def kull_leib(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(torch.distributions.Normal(mu, sig), self.N).mean()

<<<<<<< HEAD
    def reparameterize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return mu + sig*self.N.sample(mu.shape).to(mu.device)

    def forward(self, rgb_with_object: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        rgb_with_object_flat = torch.flatten(rgb_with_object, start_dim=1)
        object_mask_flat = torch.flatten(object_mask.float(), start_dim=1)
        encoder_input = torch.concat([rgb_with_object_flat, object_mask_flat], dim=1)
        features = self.feature_extractor(encoder_input)
=======
    def forward(self, object_rgb: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        object_rgb_resized = TF.resize(object_rgb, (self.model_image_size, self.model_image_size))
        object_rgb_resized_flat = torch.flatten(object_rgb_resized, start_dim=1)
        object_rgb_features = self.feature_extractor_object(object_rgb_resized_flat)

        object_mask_resized = TF.resize(object_mask.float(), (self.model_image_size, self.model_image_size))
        object_mask_resized_flat = torch.flatten(object_mask_resized, start_dim=1)
        object_mask_features = self.feature_extractor_mask(object_mask_resized_flat)

        features_concat = torch.concat([object_rgb_features, object_mask_features], dim=1)
        features = self.feature_combiner(features_concat)
>>>>>>> d726904 (WIP)
        sig = torch.exp(self.sigma_head(features))  # make it stay positive
        mu = self.mean_head(features)
        #reparameterize to find z
        z = self.reparameterize(mu, sig)
        #loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)
        return z


<<<<<<< HEAD
class DecoderFullyConnected(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
=======
class DecoderFullyConnected(Decoder):
    def __init__(self, latent_dims: int, output_image_size: int, model_image_size: int, hdim: List[int]):
>>>>>>> d726904 (WIP)
        super(DecoderFullyConnected, self).__init__()
        self.output_image_size = output_image_size
        self.model_image_size = model_image_size
        self.feature_extractor, out_features = _create_fully_connected_layers(input_dim=latent_dims, hdim=hdim[::-1])
        #self.feature_extractor_object, out_features = _create_fully_connected_layers(input_dim=latent_dims, hdim=hdim[::-1])
        self.image_head = nn.Linear(out_features, self.model_image_size**2*3)
        self.mask_head = nn.Linear(out_features, self.model_image_size**2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(z)
        #features_object = self.feature_extractor_object(z)
        object_rgb = self.image_head(features).view(-1, 3, self.model_image_size, self.model_image_size)

        #features_mask = self.feature_extractor_mask(z)
        mask_logits = self.mask_head(features).view(-1, self.model_image_size, self.model_image_size)

        object_rgb_resized = TF.resize(object_rgb, (self.output_image_size, self.output_image_size))
        mask_logits_resized = TF.resize(mask_logits, (self.output_image_size, self.output_image_size))

        return object_rgb_resized, mask_logits_resized
