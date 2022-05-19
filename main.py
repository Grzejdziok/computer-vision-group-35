from typing import Optional
import argparse
import torch
from datetime import datetime
import pytorch_lightning as pl

from data.data_generator import DataGenerator
from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.real_data_generator import RealDataGenerator
from data.datamodule import SingleItemGenerationDataModule

from models.vae_end_to_end import VAEEndToEndFullyConnected
from models.gan_fc_e2e import GANEndToEndFullyConnected
import matplotlib.pyplot as plt


VAE_FC = "vae_fc"
GAN_FC = "gan_fc"
SYNTHETIC = "synthetic"
SINGLE_ITEM_BOXES_IN_FLAT_32 = "single_item_boxes_in_flat_32"
ALL_BOXES_IN_FLAT_32 = "all_boxes_in_flat_32"


def get_data_generator(dataset_type: str) -> DataGenerator:
    if dataset_type == SYNTHETIC:
        return GaussianNoiseWithSquareSyntheticDataGenerator(image_size=(16, 16), num_samples=10000, square_size=7)
    elif dataset_type == SINGLE_ITEM_BOXES_IN_FLAT_32:
        return RealDataGenerator(resize=True, resize_dims=(32, 32), dataset_dir="dataset", single_item_box_only=True)
    elif dataset_type == ALL_BOXES_IN_FLAT_32:
        return RealDataGenerator(resize=True, resize_dims=(32, 32), dataset_dir="dataset", single_item_box_only=False)
    else:
        raise ValueError(f"Unknown type {dataset_type}")


def get_model(model_name: str, datamodule: SingleItemGenerationDataModule) -> pl.LightningModule:

    dataset_statistics = datamodule.statistics

    if model_name == VAE_FC:
        latent_dims = 512
        hidden_dims = 1024
        lr = 1e-3
        betas = (0.5, 0.999)  # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper
        return VAEEndToEndFullyConnected(latent_dims=latent_dims,
                                         s_img=dataset_statistics.image_size[0],
                                         hdim=[hidden_dims, hidden_dims, hidden_dims, hidden_dims, hidden_dims],
                                         lr=lr,
                                         betas=betas,
                                         dataset_mean=dataset_statistics.mean,
                                         dataset_std=dataset_statistics.std,
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


def main(model_name: str, dataset_type: str, batch_size: int, max_steps: Optional[int], load_weights_from: Optional[str], predict_only: bool) -> None:
    assert predict_only or max_steps is not None
    data_generator = get_data_generator(dataset_type=dataset_type)
    datamodule = SingleItemGenerationDataModule(data_generator=data_generator, batch_size=batch_size)
    datamodule.prepare_data()
    model = get_model(model_name=model_name, datamodule=datamodule)

    if load_weights_from is not None:
        model = torch.load(load_weights_from)

    if not predict_only:
        trainer = pl.Trainer(max_steps=max_steps, accelerator='gpu', devices=1, enable_checkpointing=False)
        trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader())
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
    # python main.py --model-name vae_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 100 --max-steps 30000
    # python main.py --model-name gan_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 30 --max-steps 10000
    # python main.py --model-name vae_fc --dataset-type single_item_boxes_in_flat_32 --batch-size 10 --load-weights-from vae_fc.pt --predict-only

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=[VAE_FC, GAN_FC], required=True)
    parser.add_argument("--dataset-type", choices=[SINGLE_ITEM_BOXES_IN_FLAT_32, ALL_BOXES_IN_FLAT_32], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--load-weights-from", required=False, default=None)
    parser.add_argument("--predict-only", action="store_true", default=False)
    args = parser.parse_args()

    main(model_name=args.model_name,
         dataset_type=args.dataset_type,
         load_weights_from=args.load_weights_from,
         predict_only=args.predict_only,
         batch_size=args.batch_size,
         max_steps=args.max_steps,
         )
