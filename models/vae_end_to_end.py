from typing import Tuple, List
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput
from models.vae_utils import Encoder, Decoder


class VAEEndToEnd(pl.LightningModule):
    def __init__(self, latent_dims: int, s_img: int, hdim: List[int]):
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
        reconstruction_rgb_mse_loss = F.mse_loss(
            input=image,
            target=model_targets["rgb_with_object"],
        )
        rgb_similarity_idxs = torch.sigmoid(mask_logits.detach()) < 0.5
        mask_rgb_consistency_rgb_mse_loss = F.mse_loss(
            input=image[rgb_similarity_idxs],
            target=rgb[rgb_similarity_idxs],
        )

        loss = mask_cross_entropy_loss + reconstruction_rgb_mse_loss + mask_rgb_consistency_rgb_mse_loss + self.encoder.kl
        self.log("ce_loss", mask_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("reconstruction_mse", reconstruction_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("consistency_mse", mask_rgb_consistency_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("vae_kl_loss", self.encoder.kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
