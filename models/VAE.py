from statistics import mode
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput


class Encoder(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: Tuple[int, int]):
        super(Encoder, self).__init__()
        
        self.mean_layers = nn.Sequential(
            nn.Linear(s_img*s_img*3, hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], latent_dims)
        )
        self.sigma_layers = nn.Sequential(
            nn.Linear(s_img*s_img*3, hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], latent_dims),
        )

        #distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def kull_leib(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

    def reparameterize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return mu + sig*self.N.sample(mu.shape).to(mu.device)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:

        x = rgb.flatten(start_dim=1)

        sig = self.sigma_layers(x)
        sig = torch.exp(sig) #because it has to stay positive
        mu = self.mean_layers(x)
        
        #reparameterize to find z
        z = self.reparameterize(mu, sig)

        #loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims: int, s_img: int, hdim: Tuple[int, int]):
        super(Decoder, self).__init__()
        
        self.s_img=s_img
        self.image_layers = nn.Sequential(
            nn.Linear(latent_dims, hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], s_img*s_img*3),
        )
        self.mask_layers = nn.Sequential(
            nn.Linear(latent_dims, hdim[1]),
            nn.ReLU(),
            nn.Linear(hdim[1], hdim[0]),
            nn.ReLU(),
            nn.Linear(hdim[0], s_img*s_img),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        image = self.image_layers(z)
        image = image.reshape((-1, self.s_img, self.s_img, 3))
        mask = self.mask_layers(z)
        mask = mask.reshape((-1, self.s_img, self.s_img))

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
        image, mask = self._encoder_decoder_forward(rgb)
        model_outputs = ModelOutput(rgb_with_object=image, soft_object_mask=torch.sigmoid(mask).float(),)
        return model_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        image, mask = self._encoder_decoder_forward(rgb=rgb)
        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=mask,
            target=model_targets["object_mask"].float()
        )
        rgb_mse_loss = F.mse_loss(
            input=image,
            target=model_targets["rgb_with_object"],
        )
        loss = mask_cross_entropy_loss + rgb_mse_loss + self.encoder.kl*0.1
        self.log("ce_loss", mask_cross_entropy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("mse_loss", rgb_mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_loss", self.encoder.kl*0.1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
