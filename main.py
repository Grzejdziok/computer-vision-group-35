from typing import Tuple
import argparse
import torch
import torchvision
import pytorch_lightning as pl
import os

from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.synthetic_data import SyntheticDataModule
from data.real_data_generator import RealDataGenerator
from data.real_data import RealDataModule

from models.vae_end_to_end import VAEEndToEndFullyConnected
from models.gan_fc_e2e import GANEndToEndFullyConnected
import matplotlib.pyplot as plt

from data.CVAT_reader import create_masks, create_images_1


VAE_FC = "vae_fc"
GAN_FC = "gan_fc"


def get_model(model_name: str, image_size: Tuple[int, int]) -> pl.LightningModule:
    if model_name == VAE_FC:
        latent_dims = 512
        hidden_dims = 1024
        lr = 1e-3
        betas = (0.5,
                 0.999)  # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper
        return VAEEndToEndFullyConnected(latent_dims=latent_dims, s_img=image_size[0],
                                         hdim=[hidden_dims, hidden_dims, hidden_dims, hidden_dims, hidden_dims], lr=lr,
                                         betas=betas)
    elif model_name == GAN_FC:
        noise_dim = 32
        hidden_dims_g = [1024, 1024, 1024, 1024, 1024]
        hidden_dims_d = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        lr = 1e-2
        betas = (0.5,
                 0.999)  # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper
        return GANEndToEndFullyConnected(width=image_size[0], height=image_size[1], noise_dim=noise_dim,
                                         hidden_dims_g=hidden_dims_g, hidden_dims_d=hidden_dims_d, lr=lr, betas=betas,)
    else:
        raise ValueError()


def main(model_name: str):
    rerun = False
    masks_dir = "masks"
    image_dir = "images"
    cvat_xml = "images/annotations.xml"
    image_single_dir = "images_single_object"
    datamodule_dir = "datamodule.json"
    if not os.path.exists(masks_dir) or not os.path.exists(image_single_dir) or rerun:
        create_masks(masks_dir, image_dir, cvat_xml)
        create_images_1(masks_dir, image_dir, image_single_dir)
    real_data_generator = RealDataGenerator()
    print(f"Total number of samples is: {real_data_generator._total_samples(image_single_dir)}")

    train_ratio = 0.99
    image_size = (32, 32)
    square_size = 7
    batch_size = 30
    datamodule = RealDataModule(
        train_ratio=train_ratio,
        real_data_generator=real_data_generator,
        batch_size=batch_size,
        resize=True,
        resize_dims=image_size,
        masks_dir=masks_dir,
        image_dir=image_single_dir,
        datamodule_dir=datamodule_dir
    )
    model = get_model(model_name=model_name, image_size=image_size)

    trainer = pl.Trainer(max_steps=5000, accelerator='gpu', devices=1, enable_checkpointing=False)
    trainer.fit(model=model, datamodule=datamodule)

    batch = next(iter(datamodule.predict_dataloader()))
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    rgb_gt = batch["model_input"]["rgb"].permute((0, 2, 3, 1))
    rgb_object_gt = batch["model_target"]["rgb_with_object"].permute((0, 2, 3, 1))
    mask_gt = batch["model_target"]["object_mask"]
    rgb_pred = outputs['rgb_with_object'].permute((0, 2, 3, 1))
    mask_pred = outputs['soft_object_mask']

    fig, axes = plt.subplots(nrows=5, ncols=5, sharex=False, sharey=False)
    for image_index, (ax1, ax2, ax3, ax4, ax5) in enumerate(axes):
        ax1.imshow(rgb_gt[image_index])
        ax2.imshow(rgb_object_gt[image_index])
        ax3.imshow(mask_gt[image_index])
        ax4.imshow(rgb_pred[image_index])
        ax5.imshow(mask_pred[image_index])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        if image_index == 0:
            ax1.set_title("RGB - input")
            ax2.set_title("RGB - target")
            ax3.set_title("Mask - target")
            ax4.set_title("RGB - predicted")
            ax5.set_title("Mask - predicted")
    plt.savefig("results.png")
    plt.show()


if __name__ == "__main__":
    # example usage:
    # python main.py --model-name vae_fc
    # python main.py --model-name gan_fc

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=[VAE_FC, GAN_FC], required=True)
    args = parser.parse_args()

    main(model_name=args.model_name)
