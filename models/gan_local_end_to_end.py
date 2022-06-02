import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from collections import OrderedDict
from data.training_sample import TrainingSample, ModelOutput
from typing import List, Tuple
import torchvision.transforms.functional as TF
from models.gan_local_utils import GeneratorLocal, DiscriminatorLocal
import matplotlib.pyplot as plt


class GANEndToEndFullyConnected(LightningModule):
    def __init__(
        self,
        generator: GeneratorLocal,
        discriminator: DiscriminatorLocal,
        preprocess_transform: torchvision.transforms.Normalize,
        noise_dim: int,
        lr: float,
        betas: Tuple[float, float],
    ):
        super().__init__()
        # networks
        self.preprocess_transform = preprocess_transform
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.lr = lr
        self.betas = betas
        self.e = 0

    def forward(self, batch: TrainingSample) -> ModelOutput:
        # rgb = batch["model_input"]["rgb"]
        # z = torch.normal(0., 1., (rgb.shape[0], self.hparams.noise_dim)).type_as(rgb)
        # image, mask_logits = self.generator(z, rgb)
        # soft_object_mask = torch.sigmoid(mask_logits).float()
        # model_outputs = ModelOutput(rgb_with_object=image, soft_object_mask=soft_object_mask)
        # return model_outputs
        rgb = batch["model_input"]["rgb"]
        image_height = rgb.shape[2]
        image_width = rgb.shape[3]

        z = torch.normal(0., 1., (rgb.shape[0], self.noise_dim)).type_as(rgb)
        normalized_rgb = self.preprocess_transform(rgb)
        zoomed_object_rgb, zoomed_mask_logits, boxes = self.generator(z, normalized_rgb, rgb)
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

            rgb_with_object[i, :, ymin:ymax+1, xmin:xmax+1] = TF.resize(
                img=single_zoomed_object_rgb,
                size=(ymax-ymin+1, xmax-xmin+1),
            )
            soft_object_mask[i, ymin:ymax+1, xmin:xmax+1] = TF.resize(
                img=single_zoomed_soft_object_mask.float().unsqueeze(0),
                size=(ymax-ymin+1, xmax-xmin+1),
            )[0]

        model_outputs = ModelOutput(rgb_with_object=rgb_with_object, soft_object_mask=soft_object_mask,)
        return model_outputs

    def loss(self, pred:Tuple[torch.Tensor, torch.Tensor], gt: torch.Tensor):
        mask_cross_entropy_loss = F.binary_cross_entropy(pred[1],gt)
        background_rgb_mse_loss = F.binary_cross_entropy(input=pred[0], target=gt)
        return (mask_cross_entropy_loss+background_rgb_mse_loss)/2
    
    def adversarial_loss(self, pred:torch.Tensor, gt: torch.Tensor):
        adversarial_loss = torch.nn.BCELoss()
        return adversarial_loss(pred, gt)

    def training_step(self, batch: TrainingSample, batch_idx: int, optimizer_idx: int) -> nn.Module:
        rgb = batch["model_input"]["rgb"]
        model_targets = batch["model_target"]
        zoomed_object_rgb = model_targets["zoomed_object_rgb"]
        zoomed_object_mask = model_targets["zoomed_object_mask"]
        normalized_bounding_box_xyxy = model_targets["normalized_bounding_box_xyxy"]
        for i in range(rgb.shape[0]):
            if torch.rand(1) < 0.5:
                rgb[i] = TF.hflip(rgb[i])
                zoomed_object_rgb[i] = TF.hflip(zoomed_object_rgb[i])
                zoomed_object_mask[i] = TF.hflip(zoomed_object_mask[i])
                normalized_bounding_box_xyxy[i] = torch.tensor([
                    1. - normalized_bounding_box_xyxy[i][2],
                    normalized_bounding_box_xyxy[i][1],
                    1. - normalized_bounding_box_xyxy[i][0],
                    normalized_bounding_box_xyxy[i][3],
                ],)
            if torch.rand(1) < 0.5:
                rgb[i] = TF.vflip(rgb[i])
                zoomed_object_rgb[i] = TF.vflip(zoomed_object_rgb[i])
                zoomed_object_mask[i] = TF.vflip(zoomed_object_mask[i])
                normalized_bounding_box_xyxy[i] = torch.tensor([
                    normalized_bounding_box_xyxy[i][0],
                    1. - normalized_bounding_box_xyxy[i][3],
                    normalized_bounding_box_xyxy[i][2],
                    1. - normalized_bounding_box_xyxy[i][1],
                ], )

        preprocessed_rgb = self.preprocess_transform(rgb)

        # train generator
        if optimizer_idx == 0:
            # ground truth result (ie: all fake), put on GPU
            z = torch.normal(0., 1., (rgb.shape[0], self.noise_dim)).type_as(rgb)
            predicted_zoomed_object_rgb, predicted_zoomed_object_mask_logits, predicted_boxes = self.generator(z, preprocessed_rgb, rgb)
            valid = torch.ones(predicted_zoomed_object_rgb.size(0), 1).type_as(predicted_zoomed_object_rgb)
            g_loss = self.adversarial_loss(self.discriminator(predicted_zoomed_object_rgb, predicted_zoomed_object_mask_logits, predicted_boxes), valid)
            self.log("gen_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return g_loss

        # train discriminator
        elif optimizer_idx == 1:
            z = torch.normal(0., 1., (rgb.shape[0], self.noise_dim)).type_as(rgb)
            predicted_zoomed_object_rgb, predicted_zoomed_object_mask_logits, predicted_boxes = self.generator(z, preprocessed_rgb, rgb) #detached by default apparently

            valid = torch.ones(predicted_zoomed_object_rgb.size(0), 1).type_as(predicted_zoomed_object_rgb)
            real_loss = self.adversarial_loss(self.discriminator(zoomed_object_rgb, zoomed_object_mask, normalized_bounding_box_xyxy), valid)

            # plt.imshow(zoomed_object_rgb[0].permute(1,2,0).cpu().numpy())
            # plt.imshow(zoomed_object_mask[0].cpu().numpy())
            fake = torch.zeros(predicted_zoomed_object_rgb.size(0), 1).type_as(predicted_zoomed_object_rgb)
            fake_loss = self.adversarial_loss(self.discriminator(predicted_zoomed_object_rgb, predicted_zoomed_object_mask_logits, predicted_boxes), fake)
            # plt.imshow(predicted_zoomed_object_rgb[0].permute(1,2,0).cpu().numpy())
            # plt.imshow(predicted_zoomed_object_mask_logits[0].cpu().numpy())

            with torch.no_grad():
                outputs = self(batch)
                image = outputs['rgb_with_object']

            self.e+=1
            plt.axis('off')
            plt.imsave(f'/home/svoloshyn/GAN/{self.e}.png', (image[0].permute(1,2,0).cpu().numpy()))
            # average
            d_loss = (real_loss + fake_loss) / 2
            self.log("real_loss", real_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("fake_loss", fake_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("ms", torch.mean(torch.square(predicted_zoomed_object_rgb)), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return d_loss
        else:
            raise ValueError

    def configure_optimizers(self):
        lr = self.lr
        betas = self.betas

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []
