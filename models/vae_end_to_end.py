from typing import Tuple, List, Optional
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput
from models.vae_utils import EncoderFullyConnected, DecoderFullyConnected


class VAEEndToEndFullyConnected(pl.LightningModule):
    def __init__(self,
                 latent_dims: int,
                 s_img: int,
                 hdim: List[int],
                 lr: float,
                 betas: Tuple[float, float],
                 dataset_mean: Optional[Tuple[float, float, float]],
                 dataset_std: Optional[Tuple[float, float, float]],
                 ):
        super().__init__()
        self.latent_dims = latent_dims
        self.lr = lr
        self.betas = betas
        self.encoder = EncoderFullyConnected(latent_dims, s_img, hdim)
        self.decoder = DecoderFullyConnected(latent_dims, s_img, hdim)
        self.dataset_mean = torch.tensor(dataset_mean or (0., 0., 0.))
        self.dataset_std = torch.tensor(dataset_std or (1., 1., 1.))
        self.preprocess_transform = torchvision.transforms.Normalize(
            mean=self.dataset_mean,
            std=self.dataset_std,
        )

    def forward(self, batch: TrainingSample) -> ModelOutput:
        rgb = batch["model_input"]["rgb"]
        z = torch.normal(0., 1., (rgb.shape[0], self.latent_dims))
        normalized_rgb = self.preprocess_transform(rgb)
        image, mask_logits = self.decoder(z, normalized_rgb, rgb)
        soft_object_mask = torch.sigmoid(mask_logits).float()
        model_outputs = ModelOutput(rgb_with_object=image, soft_object_mask=soft_object_mask,)
        return model_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        rgb_with_object = model_targets["rgb_with_object"]
        object_mask = model_targets["object_mask"]

        for i in range(rgb.shape[0]):
            if torch.rand(1) < 0.5:
                rgb[i] = TF.hflip(rgb[i])
                rgb_with_object[i] = TF.hflip(rgb_with_object[i])
                object_mask[i] = TF.hflip(object_mask[i])
            if torch.rand(1) < 0.5:
                rgb[i] = TF.vflip(rgb[i])
                rgb_with_object[i] = TF.vflip(rgb_with_object[i])
                object_mask[i] = TF.vflip(object_mask[i])

        preprocessed_rgb_with_object = self.preprocess_transform(rgb_with_object)
        z = self.encoder(preprocessed_rgb_with_object, object_mask)

        preprocessed_rgb = self.preprocess_transform(rgb)
        predicted_rgb_with_object, predicted_object_mask_logits = self.decoder(z, preprocessed_rgb, rgb)
        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=predicted_object_mask_logits,
            target=object_mask.float(),
        )
        background_rgb_mse_loss = F.mse_loss(
            input=predicted_rgb_with_object,
            target=rgb_with_object,
        )
        object_rgb_mse_loss = 0.
        for single_predicted_rgb_with_object, single_rgb_with_object, single_object_mask in zip(predicted_rgb_with_object, rgb_with_object, object_mask):
            object_rgb_mse_loss += F.mse_loss(
                input=single_predicted_rgb_with_object[:, single_object_mask],
                target=single_rgb_with_object[:, single_object_mask],
            )
        object_rgb_mse_loss /= predicted_rgb_with_object.shape[0]
        loss = 10. * mask_cross_entropy_loss + self.encoder.kl + object_rgb_mse_loss + background_rgb_mse_loss
        self.log("ce_loss", mask_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("background_mse", background_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("object_mse", object_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("vae_kl_loss", self.encoder.kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

