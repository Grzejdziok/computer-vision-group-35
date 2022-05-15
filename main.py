import torch
import torchvision
import pytorch_lightning as pl
import os

from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.synthetic_data import SyntheticDataModule
from data.real_data_generator import RealDataGenerator
from data.real_data import RealDataModule

from models.vae_end_to_end import VAEEndToEndFullyConnected
from models.gan_end_to_end import GANEndToEnd
import matplotlib.pyplot as plt

from data.CVAT_reader import create_masks, create_images_1


if __name__ == "__main__":
    

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

    # num_train_samples = 500
    # num_val_samples = 150
    train_ratio = 0.75
    image_size = (32, 32)
    square_size = 7
    batch_size = 100
    datamodule = RealDataModule(
        train_ratio = train_ratio,
        real_data_generator=real_data_generator,
        batch_size=batch_size,
        resize=True,
        resize_dims=image_size,
        masks_dir=masks_dir,
        image_dir=image_single_dir,
        datamodule_dir=datamodule_dir
    )

    latent_dims = 512
    hidden_dims = 1024
    lr = 1e-3
    betas = (0.5, 0.999) # coefficients used for computing running averages of gradient and its square for Adam - from GauGAN paper
    model = VAEEndToEndFullyConnected(latent_dims=latent_dims, s_img=image_size[0], hdim=[hidden_dims, hidden_dims, hidden_dims, hidden_dims, hidden_dims], lr=lr, betas=betas)
    # model = GANEndToEnd(width=image_size[0], height=image_size[1], latent_dim=latent_dims, hidden_dim=hidden_dims, lr=lr, betas=betas, batch_size=batch_size)
    
    trainer = pl.Trainer(max_steps=60, accelerator='gpu', devices=1, enable_checkpointing=False)
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

