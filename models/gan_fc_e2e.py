import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from models.gan_fc_utils import GeneratorFullyConnected, DiscriminatorFullyConnected
from collections import OrderedDict
from data.training_sample import TrainingSample, ModelOutput
from typing import List, Tuple


class GANEndToEndFullyConnected(LightningModule):
    def __init__(
        self,
        width: int,
        height: int,
        noise_dim: int,
        hidden_dims_g: List[int],
        hidden_dims_d: List[int],
        lr: float,
        betas: Tuple[float, float],
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (3, width, height)
        self.generator = GeneratorFullyConnected(noise_dim=self.hparams.noise_dim, hidden_dims=self.hparams.hidden_dims_g, img_shape=data_shape)
        self.discriminator = DiscriminatorFullyConnected(hidden_dims=self.hparams.hidden_dims_d, img_shape=data_shape)

    def forward(self, batch: TrainingSample) -> ModelOutput:
        rgb = batch["model_input"]["rgb"]
        z = torch.normal(0., 1., (rgb.shape[0], self.hparams.noise_dim)).type_as(rgb)
        image, mask_logits = self.generator(z, rgb)
        soft_object_mask = torch.sigmoid(mask_logits).float()
        model_outputs = ModelOutput(rgb_with_object=image, soft_object_mask=soft_object_mask)
        return model_outputs

    #pred[0] is rgb
    def loss(self, pred:Tuple[torch.Tensor, torch.Tensor], gt: torch.Tensor):
        mask_cross_entropy_loss = F.binary_cross_entropy_with_logits(pred[1],gt)
        background_rgb_mse_loss = F.mse_loss(input=pred[0], target=gt)
        return (mask_cross_entropy_loss+background_rgb_mse_loss)/2
    
    def adversarial_loss(self, pred:Tuple[torch.Tensor, torch.Tensor], gt: torch.Tensor):
        adversarial_loss = torch.nn.BCELoss()
        rgb_loss = adversarial_loss(pred[0], gt)
        mask_loss = adversarial_loss(pred[1], gt)
        return (rgb_loss+mask_loss)/2

    def training_step(self, batch: TrainingSample, batch_idx: int, optimizer_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        rgb_with_object = model_targets["rgb_with_object"]
        object_mask = model_targets["object_mask"].float()

        # train generator
        if optimizer_idx == 0:
            # ground truth result    (ie: all fake), put on GPU
            valid = torch.ones(rgb_with_object.size(0), 1).type_as(rgb_with_object)
            g_loss = self.adversarial_loss(self.discriminator(self(batch)), valid)
            self.log("gen_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(rgb_with_object.size(0), 1).type_as(rgb_with_object)
            real_loss = self.adversarial_loss(self.discriminator(batch), valid)

            fake = torch.zeros(rgb_with_object.size(0), 1).type_as(rgb_with_object)
            fake_loss = self.adversarial_loss(self.discriminator(self(batch)), fake)

            # average
            d_loss = (real_loss + fake_loss) / 2
            self.log("real_loss", real_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("fake_loss", fake_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.betas

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []
