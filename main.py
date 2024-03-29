import os
import shutil
from typing import Optional
import argparse
import torch
from datetime import datetime
import pytorch_lightning as pl
import torchvision

from data.data_generator import DataGenerator
from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.real_data_generator import RealDataGenerator
from data.datamodule import SingleItemGenerationDataModule

from models.vae_global_end_to_end import VAEGlobalEndToEnd
from models.vae_global_utils import EncoderGlobalFullyConnected, DecoderGlobalFullyConnected
from models.vae_local_end_to_end import VAELocalEndToEnd
from models.vae_local_utils import EncoderLocalFullyConnected, DecoderLocalFullyConnected, EncoderLocalConvolutional, DecoderLocalConvolutional
from models.gan_fc_e2e import GANEndToEndFullyConnected

from callbacks.loss_reporter_callback import LossReporterCallback

import matplotlib.pyplot as plt


VAE_GLOBAL_FC_32 = "vae_global_fc_32"
VAE_GLOBAL_FC_64 = "vae_global_fc_64"
VAE_LOCAL_FC_32 = "vae_local_fc_32"
VAE_LOCAL_FC_64 = "vae_local_fc_64"
VAE_LOCAL_CONV_4x = "vae_local_conv_4x"
GAN_FC = "gan_fc"
SYNTHETIC = "synthetic"
SINGLE_ITEM_BOXES_IN_FLAT_32 = "single_item_boxes_in_flat_32"
SINGLE_ITEM_BOXES_IN_FLAT_128 = "single_item_boxes_in_flat_128"
SINGLE_ITEM_BOXES_IN_FLAT_256 = "single_item_boxes_in_flat_256"
ALL_BOXES_IN_FLAT_32 = "all_boxes_in_flat_32"
ALL_BOXES_IN_FLAT_256 = "all_boxes_in_flat_256"


def get_data_generator(dataset_type: str) -> DataGenerator:
    if dataset_type == SYNTHETIC:
        return GaussianNoiseWithSquareSyntheticDataGenerator(image_size=(16, 16), num_samples=10000, square_size=7)
    elif dataset_type == SINGLE_ITEM_BOXES_IN_FLAT_32:
        return RealDataGenerator(resize=True, resize_dims=(32, 32), dataset_dir="dataset", single_item_box_only=True)
    elif dataset_type == SINGLE_ITEM_BOXES_IN_FLAT_128:
        return RealDataGenerator(resize=True, resize_dims=(128, 128), dataset_dir="dataset", single_item_box_only=True)
    elif dataset_type == SINGLE_ITEM_BOXES_IN_FLAT_256:
        return RealDataGenerator(resize=True, resize_dims=(256, 256), dataset_dir="dataset", single_item_box_only=True)
    elif dataset_type == ALL_BOXES_IN_FLAT_32:
        return RealDataGenerator(resize=True, resize_dims=(32, 32), dataset_dir="dataset", single_item_box_only=False)
    elif dataset_type == ALL_BOXES_IN_FLAT_256:
        return RealDataGenerator(resize=True, resize_dims=(256, 256), dataset_dir="dataset", single_item_box_only=False)
    else:
        raise ValueError(f"Unknown type {dataset_type}")


def get_model(model_name: str, datamodule: SingleItemGenerationDataModule, width: int) -> pl.LightningModule:

    dataset_statistics = datamodule.statistics

    if model_name in [VAE_GLOBAL_FC_32, VAE_GLOBAL_FC_64]:
        latent_dims = width
        hidden_dims = 5 * [width]
        model_image_size = 32 if model_name == VAE_GLOBAL_FC_32 else 64
        lr = 1e-3
        betas = (0.5, 0.999)  # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper

        encoder = EncoderGlobalFullyConnected(
            latent_dims=latent_dims,
            input_image_size=model_image_size,
            hdim=hidden_dims,
        )
        decoder = DecoderGlobalFullyConnected(
            latent_dims=latent_dims,
            model_output_image_size=model_image_size,
            output_image_size=dataset_statistics.image_size[0],
            hdim=hidden_dims,
        )
        preprocess_transform = torchvision.transforms.Normalize(
            mean=dataset_statistics.mean,
            std=dataset_statistics.std,
        )
        return VAEGlobalEndToEnd(
            encoder=encoder,
            decoder=decoder,
            preprocess_transform=preprocess_transform,
            lr=lr,
            betas=betas,
        )
    elif model_name in [VAE_LOCAL_FC_32, VAE_LOCAL_FC_64]:
        latent_dims = width
        hidden_dims = 5 * [width]
        model_image_size = 32 if model_name == VAE_LOCAL_FC_32 else 64
        lr = 1e-3
        betas = (0.9, 0.999)
        encoder = EncoderLocalFullyConnected(
            latent_dims=latent_dims,
            input_image_size=model_image_size,
            hdim=hidden_dims,
        )
        decoder = DecoderLocalFullyConnected(
            latent_dims=latent_dims,
            model_output_image_size=model_image_size,
            output_image_size=dataset_statistics.image_size[0],
            hdim=hidden_dims,
            rgb_embedding_dims=0,
        )
        preprocess_transform = torchvision.transforms.Normalize(
            mean=dataset_statistics.mean,
            std=dataset_statistics.std,
        )
        return VAELocalEndToEnd(
            encoder=encoder,
            decoder=decoder,
            preprocess_transform=preprocess_transform,
            lr=lr,
            betas=betas,
        )
    elif model_name == VAE_LOCAL_CONV_4x:
        latent_dims = 256
        hidden_channels_per_block = [[32, 32], [64, 64], [128, 128], [256, 256]]
        lr = 1e-3
        betas = (0.9, 0.999)
        encoder = EncoderLocalConvolutional(
            hidden_channels_per_block=hidden_channels_per_block,
            latent_dims=latent_dims,
        )
        decoder = DecoderLocalConvolutional(
            latent_dims=latent_dims,
            hidden_channels_per_block=hidden_channels_per_block,
            output_image_size=dataset_statistics.image_size[0],
        )
        preprocess_transform = torchvision.transforms.Normalize(
            mean=dataset_statistics.mean,
            std=dataset_statistics.std,
        )
        return VAELocalEndToEnd(
            encoder=encoder,
            decoder=decoder,
            preprocess_transform=preprocess_transform,
            lr=lr,
            betas=betas,
        )
    elif model_name == GAN_FC:
        noise_dim = 32
        hidden_dims_g = [1024, 1024, 1024, 1024, 1024]
        hidden_dims_d = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        lr = 1e-2
        betas = (0.5, 0.999)  # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper
        return GANEndToEndFullyConnected(width=dataset_statistics.image_size[0],
                                         height=dataset_statistics.image_size[1],
                                         noise_dim=noise_dim,
                                         hidden_dims_g=hidden_dims_g,
                                         hidden_dims_d=hidden_dims_d,
                                         lr=lr,
                                         betas=betas,)
    else:
        raise ValueError()


def main(model_name: str,
         dataset_type: str,
         batch_size: int,
         width: int,
         max_steps: Optional[int],
         load_weights_from: Optional[str],
         predict_only: bool,
         test_ratio: float,
         generate_survey_images: bool,
         ) -> None:
    assert predict_only or max_steps is not None
    data_generator = get_data_generator(dataset_type=dataset_type)
    datamodule = SingleItemGenerationDataModule(data_generator=data_generator, batch_size=batch_size, test_ratio=test_ratio)
    datamodule.prepare_data()
    model = get_model(model_name=model_name, datamodule=datamodule, width=width)

    if load_weights_from is not None:
        model = torch.load(load_weights_from)

    if not predict_only:
        loss_reporter_callback = LossReporterCallback()
        trainer = pl.Trainer(max_steps=max_steps, accelerator='gpu', devices=1, enable_checkpointing=False, callbacks=[loss_reporter_callback], num_sanity_val_steps=0,)
        trainer.fit(model=model, datamodule=datamodule)
        torch.save(model, f"{model_name}_{str(datetime.now()).replace(':', '_')}.pt")

        # plot losses
        plt.clf()
        loss_reporter_callback.plot_and_save_losses("losses.png")
        plt.show()

    # plot predictions
    num_predictions = 5
    model.eval()
    fig, axes = plt.subplots(nrows=num_predictions, ncols=5, sharex=False, sharey=False)
    for i, (batch, axes_cols) in enumerate(zip(datamodule.predict_dataloader(), axes)):

        with torch.no_grad():
            outputs = model(batch)

        rgb_gt = batch["model_input"]["rgb"][0].permute((1, 2, 0))  # make channels last
        rgb_object_gt = batch["model_target"]["rgb_with_object"][0].permute((1, 2, 0))  # make channels last
        mask_gt = batch["model_target"]["object_mask"][0]
        rgb_pred = outputs['rgb_with_object'][0].permute((1, 2, 0))  # make channels last
        mask_pred = outputs['soft_object_mask'][0]

        for ax in axes_cols:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        titles = ["RGB - input", "RGB - target", "Mask - target", "RGB - predicted", "Mask - predicted"]
        imgs = [rgb_gt, rgb_object_gt, mask_gt, rgb_pred, mask_pred]
        for ax, title, img in zip(axes_cols, titles, imgs):
            if i == 0:
                ax.set_title(title)
            ax.imshow(img)

    plt.savefig("visualizations.png")
    plt.show()
    if generate_survey_images:
        directory = "survey_images"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        model.eval()
        for i, batch in enumerate(datamodule.predict_dataloader()):
            assert batch["model_target"]["rgb_with_object"].shape[0] == 1
            with torch.no_grad():
                outputs = model(batch)

            path_real = os.path.join(directory, f"{i}_real.png")
            rgb_object_gt = batch["model_target"]["rgb_with_object"][0]
            torchvision.utils.save_image(rgb_object_gt, path_real)

            path_fake = os.path.join(directory, f"{i}_fake.png")
            rgb_pred = outputs['rgb_with_object'][0]
            torchvision.utils.save_image(rgb_pred, path_fake)


if __name__ == "__main__":
    # example usage:
    # python main.py --model-name vae_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 100 --max-steps 30000
    # python main.py --model-name gan_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 30 --max-steps 10000
    # python main.py --model-name vae_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 10 --load-weights-from vae_fc.pt --predict-only
    #
    # survey usage: python main.py --model-name vae_local_fc_64 --dataset-type all_boxes_in_flat_256 --batch-size 100 --max-steps 30000 --width 1024 --generate-survey-images

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=[VAE_GLOBAL_FC_32, VAE_GLOBAL_FC_64, VAE_LOCAL_FC_32, VAE_LOCAL_FC_64, VAE_LOCAL_CONV_4x, GAN_FC], required=True)
    parser.add_argument("--dataset-type", choices=[SINGLE_ITEM_BOXES_IN_FLAT_32, SINGLE_ITEM_BOXES_IN_FLAT_128, SINGLE_ITEM_BOXES_IN_FLAT_256, ALL_BOXES_IN_FLAT_32, ALL_BOXES_IN_FLAT_256], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--load-weights-from", required=False, default=None)
    parser.add_argument("--predict-only", action="store_true", default=False)
    parser.add_argument("--generate-survey-images", action="store_true", default=False)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    main(model_name=args.model_name,
         dataset_type=args.dataset_type,
         load_weights_from=args.load_weights_from,
         predict_only=args.predict_only,
         batch_size=args.batch_size,
         max_steps=args.max_steps,
         test_ratio=args.test_ratio,
         width=args.width,
         generate_survey_images=args.generate_survey_images,
         )
