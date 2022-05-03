from typing import Tuple, List
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput


class Encoder(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
        super(Encoder, self).__init__()

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


class Decoder(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
        super(Decoder, self).__init__()

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


class LitVAE(pl.LightningModule):
    def __init__(self, latent_dims, s_img, hdim):
        super().__init__()
        self.encoder = Encoder(latent_dims, s_img, hdim)
        self.decoder = Decoder(latent_dims, s_img, hdim)

    def _encoder_decoder_forward(self, rgb: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        z = self.encoder(rgb)
        image, mask = self.decoder(z)
        return image, mask

    def forward(self, batch: TrainingSample) -> ModelOutput:
        rgb = batch["model_input"]["rgb"]
        image, mask_logits = self._encoder_decoder_forward(rgb)
        soft_object_mask = torch.sigmoid(mask_logits).float()
        model_outputs = ModelOutput(rgb_with_object=image, soft_object_mask=soft_object_mask,)
        return model_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        image, mask_logits = self._encoder_decoder_forward(rgb=rgb)
        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=mask_logits,
            target=model_targets["object_mask"].float(),
        )
        rgb_mse_loss = F.mse_loss(
            input=image,
            target=model_targets["rgb_with_object"],
        )
        loss = mask_cross_entropy_loss + rgb_mse_loss + self.encoder.kl
        self.log("ce_loss", mask_cross_entropy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("mse_loss", rgb_mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_loss", self.encoder.kl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
