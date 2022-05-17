from typing import Tuple, Optional
import argparse
import torch
from datetime import datetime
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


def main(model_name: str, load_weights_from: Optional[str], predict_only: bool) -> None:
    dataset_dir = "dataset"
    real_data_generator = RealDataGenerator()

    image_size = (32, 32)
    square_size = 7
    batch_size = 100
    datamodule = RealDataModule(
        real_data_generator=real_data_generator,
        batch_size=batch_size,
        resize=True,
        resize_dims=image_size,
        dataset_dir=dataset_dir,
        single_item_box_only=True,
    )
    datamodule.prepare_data()
    model = get_model(model_name=model_name, image_size=image_size)

    if load_weights_from is not None:
        model = torch.load(load_weights_from)

    if not predict_only:
        trainer = pl.Trainer(max_steps=30000, accelerator='gpu', devices=1, enable_checkpointing=False)
        trainer.fit(model=model, datamodule=datamodule)
        torch.save(model, f"{model_name}_{datetime.now()}.pt")

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
    # python main.py --model-name vae_fc --load-weights-from vae_fc.pt --predict-only

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=[VAE_FC, GAN_FC], required=True)
    parser.add_argument("--load-weights-from", required=False, default=None)
    parser.add_argument("--predict-only", action="store_true", default=False)
    args = parser.parse_args()

    main(model_name=args.model_name, load_weights_from=args.load_weights_from, predict_only=args.predict_only)
