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

    def loss(self, pred:Tuple[torch.Tensor, torch.Tensor], gt: torch.Tensor):
        rgb_loss = F.binary_cross_entropy(pred[0], gt)
        mask_loss = F.binary_cross_entropy(pred[1], gt)
        return (rgb_loss+mask_loss)/2


    def training_step(self, batch: TrainingSample, batch_idx: int, optimizer_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        rgb_with_object = model_targets["rgb_with_object"]
        object_mask = model_targets["object_mask"].float()


        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(batch)

            # ground truth result (ie: all fake), put on GPU
            valid = torch.ones(rgb_with_object.size(0), 1).type_as(rgb_with_object)

            # adversarial loss is binary cross-entropy
            g_loss = self.loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # real
            valid = torch.ones(rgb_with_object.size(0), 1).type_as(rgb_with_object)
            real_loss = self.loss(self.discriminator(batch), valid)

            # fake
            noise_image = torch.randn(rgb_with_object.shape).type_as(rgb_with_object)
            noise_mask = torch.randn(object_mask.shape).type_as(object_mask)

            noise_dict = ModelOutput(rgb_with_object=noise_image, soft_object_mask=noise_mask)
            fake = torch.zeros(rgb_with_object.size(0), 1).type_as(rgb_with_object)
            fake_loss = self.loss(self.discriminator(noise_dict), fake)

            # average
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.betas

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []
