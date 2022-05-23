from typing import Tuple
import torch
import torchvision
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput
from models.vae_utils import EncoderFullyConnected, DecoderFullyConnected


class VAEEndToEnd(pl.LightningModule):
    def __init__(self,
                 encoder: EncoderFullyConnected,
                 decoder: DecoderFullyConnected,
                 preprocess_transform: torchvision.transforms.Normalize,
                 lr: float,
                 betas: Tuple[float, float],
                 ):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.encoder = encoder
        self.decoder = decoder
        self.preprocess_transform = preprocess_transform

    def forward(self, batch: TrainingSample) -> ModelOutput:
        rgb = batch["model_input"]["rgb"]
        img_height = rgb.shape[2]
        img_width = rgb.shape[3]

        z = self.encoder.sample(num_samples=rgb.shape[0], device=rgb.device)
        predicted_object_rgb, mask_logits = self.decoder(z)
        predicted_soft_object_mask = TF.resize(torch.sigmoid(mask_logits).float(), (img_height, img_width)).float()

        predicted_rgb_with_object = rgb.clone()
        for single_rgb_with_object, single_soft_object_mask, single_object in \
                zip(predicted_rgb_with_object, predicted_soft_object_mask, predicted_object_rgb):
            single_object_mask = single_soft_object_mask > 0.5
            if single_object_mask.sum() > 0:
                single_box = masks_to_boxes(single_object_mask.unsqueeze(0)).int()[0]
                xmin, ymin, xmax, ymax = single_box
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                single_predicted_object_resized = TF.resize(single_object, (h, w))
                single_predicted_object_padded = TF.pad(single_predicted_object_resized, (xmin, ymin, img_width - xmax - 1, img_height - ymax - 1))
                single_rgb_with_object[:] = single_predicted_object_padded * single_object_mask.float() \
                                            + single_rgb_with_object[:] * (1. - single_object_mask.float())

        model_outputs = ModelOutput(
            rgb_with_object=predicted_rgb_with_object,
            soft_object_mask=predicted_soft_object_mask,
        )
        return model_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        rgb_with_object = model_targets["rgb_with_object"]
        object_mask = model_targets["object_mask"]
        object_rgb = model_targets["object_rgb"]

        for i in range(rgb.shape[0]):
            if torch.rand(1) < 0.5:
                rgb[i] = TF.hflip(rgb[i])
                rgb_with_object[i] = TF.hflip(rgb_with_object[i])
                object_mask[i] = TF.hflip(object_mask[i])
                object_rgb[i] = TF.hflip(object_rgb[i])
            if torch.rand(1) < 0.5:
                rgb[i] = TF.vflip(rgb[i])
                rgb_with_object[i] = TF.vflip(rgb_with_object[i])
                object_mask[i] = TF.vflip(object_mask[i])
                object_rgb[i] = TF.vflip(object_rgb[i])

        preprocessed_object_rgb = self.preprocess_transform(object_rgb)
        z = self.encoder(preprocessed_object_rgb, object_mask)
        predicted_object_rgb, predicted_object_mask_logits = self.decoder(z)

        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=predicted_object_mask_logits,
            target=object_mask.float(),
        )

        background_rgb_mse_loss = 0.
        object_rgb_mse_loss = 0.
        predicted_soft_object_mask = torch.sigmoid(predicted_object_mask_logits).detach()
        for single_rgb, single_rgb_with_object, single_object_mask, single_object_rgb, single_predicted_soft_object_mask, single_predicted_object_rgb in \
                zip(rgb, rgb_with_object, object_mask, object_rgb, predicted_soft_object_mask, predicted_object_rgb):

            single_predicted_object_mask = single_predicted_soft_object_mask > 0.5
            single_predicted_object = torch.zeros_like(single_rgb)
            if single_predicted_object_mask.sum() > 0:
                single_box = masks_to_boxes(single_predicted_object_mask.unsqueeze(0)).int()[0]
                xmin, ymin, xmax, ymax = single_box
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                img_height = single_rgb.shape[1]
                img_width = single_rgb.shape[2]
                single_predicted_object_resized = TF.resize(single_predicted_object_rgb, (h, w))
                single_predicted_object_padded = TF.pad(single_predicted_object_resized, (xmin, ymin, img_width - xmax - 1, img_height - ymax - 1))
                single_predicted_object = single_predicted_object_padded
            single_predicted_rgb_with_object = single_rgb * (1. - single_predicted_object_mask.float()) \
                                               + single_predicted_object * single_predicted_object_mask.float()

            background_rgb_mse_loss += F.mse_loss(
                input=single_predicted_rgb_with_object,
                target=single_rgb_with_object,
            )
            object_rgb_mse_loss += F.mse_loss(
                input=single_predicted_rgb_with_object[:, single_object_mask],
                target=single_rgb_with_object[:, single_object_mask],
            )
            object_rgb_mse_loss += F.mse_loss(
                input=single_predicted_object_rgb,
                target=single_object_rgb,
            )
        object_rgb_mse_loss /= rgb.shape[0]
        background_rgb_mse_loss /= rgb.shape[0]
        loss = 10. * mask_cross_entropy_loss + self.encoder.kl + object_rgb_mse_loss #+ background_rgb_mse_loss
        self.log("ce_loss", mask_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("background_mse", background_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("object_mse", object_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("vae_kl_loss", self.encoder.kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

