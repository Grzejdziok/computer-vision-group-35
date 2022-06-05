from typing import Tuple
import torch
import torchvision
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torch.nn.functional as F
from data.training_sample import TrainingSample, ModelOutput
from models.vae_local_utils import EncoderLocal, DecoderLocal


class VAELocalEndToEnd(pl.LightningModule):
    def __init__(self,
                 encoder: EncoderLocal,
                 decoder: DecoderLocal,
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
        image_height = rgb.shape[2]
        image_width = rgb.shape[3]

        z = self.encoder.sample(num_samples=rgb.shape[0], device=rgb.device)
        normalized_rgb = self.preprocess_transform(rgb)
        zoomed_object_rgb, zoomed_mask_logits, boxes = self.decoder(z, normalized_rgb, rgb)
        zoomed_soft_object_mask = torch.sigmoid(zoomed_mask_logits)
        soft_object_mask = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).float()
        rgb_with_object = rgb.clone()
        for i, (single_zoomed_object_rgb, single_zoomed_soft_object_mask, single_box) in enumerate(zip(zoomed_object_rgb, zoomed_soft_object_mask, boxes)):
            box_int = single_box.clone()
            box_int[0] *= image_width
            box_int[1] *= image_height
            box_int[2] *= image_width
            box_int[3] *= image_height
            box_int = box_int.int()
            xmin, ymin, xmax, ymax = box_int
            xmin = torch.clamp(xmin, 0, image_width-1)
            xmax = torch.clamp(xmax, 0, image_width-1)
            ymin = torch.clamp(ymin, 0, image_height-1)
            ymax = torch.clamp(ymax, 0, image_height-1)

            resized_soft_object_mask = TF.resize(
                img=single_zoomed_soft_object_mask.float().unsqueeze(0),
                size=(ymax-ymin+1, xmax-xmin+1),
            )[0]
            soft_object_mask[i, ymin:ymax+1, xmin:xmax+1] = resized_soft_object_mask

            rgb_with_object[i, :, ymin:ymax+1, xmin:xmax+1] *= (1. - resized_soft_object_mask)
            rgb_with_object[i, :, ymin:ymax+1, xmin:xmax+1] += resized_soft_object_mask * TF.resize(
                img=single_zoomed_object_rgb,
                size=(ymax-ymin+1, xmax-xmin+1),
            )

        model_outputs = ModelOutput(rgb_with_object=rgb_with_object, soft_object_mask=soft_object_mask,)
        return model_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-5)
        return optimizer

    def _inner_training_step(self, batch: TrainingSample, prefix_metrics: str, prog_bar: bool) -> torch.Tensor:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        zoomed_object_rgb = model_targets["zoomed_object_rgb"]
        zoomed_object_mask = model_targets["zoomed_object_mask"]
        normalized_bounding_box_xyxy = model_targets["normalized_bounding_box_xyxy"]

        z = self.encoder(normalized_bounding_box_xyxy, zoomed_object_mask, zoomed_object_rgb)
        preprocessed_rgb = self.preprocess_transform(rgb)
        predicted_zoomed_object_rgb, predicted_zoomed_object_mask_logits, predicted_boxes = self.decoder(z, preprocessed_rgb, rgb)
        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=predicted_zoomed_object_mask_logits,
            target=zoomed_object_mask.float(),
        )
        object_rgb_mse_loss = F.mse_loss(
            input=predicted_zoomed_object_rgb,
            target=zoomed_object_rgb,
        )
        box_loss = F.smooth_l1_loss(
            input=predicted_boxes,
            target=normalized_bounding_box_xyxy,
        )
        loss = mask_cross_entropy_loss + self.encoder.kl + 10. * object_rgb_mse_loss + 10. * box_loss

        self.log("_".join([prefix_metrics, "ce_loss"]), mask_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=prog_bar, logger=True)
        self.log("_".join([prefix_metrics, "box_loss"]), box_loss, on_step=False, on_epoch=True, prog_bar=prog_bar, logger=True)
        self.log("_".join([prefix_metrics, "object_mse"]), object_rgb_mse_loss, on_step=False, on_epoch=True, prog_bar=prog_bar, logger=True)
        self.log("_".join([prefix_metrics, "vae_kl_loss"]), self.encoder.kl, on_step=False, on_epoch=True, prog_bar=prog_bar, logger=True)
        return loss

    def training_step(self, batch: TrainingSample, batch_idx: int) -> torch.Tensor:
        loss = self._inner_training_step(batch, "train", prog_bar=True)
        return loss

    def validation_step(self, batch: TrainingSample, batch_idx: int) -> torch.Tensor:
        val_loss = self._inner_training_step(batch, prefix_metrics="val", prog_bar=False)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
