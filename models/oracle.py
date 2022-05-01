from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from data.training_sample import TrainingSample, ModelOutput, ModelInput


class OracleModel(pl.LightningModule):
    # a dummy predictor which always returns ground-truth data

    def forward(self, batch: TrainingSample) -> ModelOutput:
        model_targets = batch["model_target"]
        model_outputs = ModelOutput(
            rgb_with_object=model_targets["rgb_with_object"].clone(),
            soft_object_mask=model_targets["object_mask"].float().clone(),
        )
        return model_outputs

    def training_step(self, batch: TrainingSample, batch_idx: int) -> nn.Module:
        model_targets = batch["model_target"]
        model_outputs = self(batch=batch)
        mask_cross_entropy_loss = F.binary_cross_entropy(
            input=model_outputs["soft_object_mask"],
            target=model_targets["object_mask"].float(),
        )
        rgb_mse_loss = F.mse_loss(
            input=model_outputs["rgb_with_object"],
            target=model_targets["rgb_with_object"],
        )
        loss = mask_cross_entropy_loss + rgb_mse_loss
        loss.requires_grad = True
        return loss

    def test_step(self, batch: TrainingSample, batch_idx: int) -> Tuple[ModelInput, ModelOutput]:
        model_inputs = batch["model_input"]
        model_outputs = self(batch, batch_idx)
        return model_inputs, model_outputs

    def predict_step(self, batch: TrainingSample, batch_idx: int, dataloader_idx: int = 0) -> Tuple[TrainingSample, ModelOutput]:
        model_outputs = self(batch, batch_idx)
        return batch, model_outputs

    def configure_optimizers(self) -> None:
        return None
