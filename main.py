import torch
import pytorch_lightning as pl

from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.synthetic_data import SyntheticDataModule

from models.vae_end_to_end import VAEEndToEndFullyConnected
import matplotlib.pyplot as plt


if __name__ == "__main__":

    num_train_samples = 100000
    num_val_samples = 1000
    image_size = (16, 16)
    square_size = 7
    batch_size = 1000

    synthetic_data_generator = GaussianNoiseWithSquareSyntheticDataGenerator(image_size=image_size, square_size=square_size)
    datamodule = SyntheticDataModule(
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        synthetic_data_generator=synthetic_data_generator,
        batch_size=batch_size,
    )
    datamodule.prepare_data()

    latent_dims = 256
    hidden_dims = 1024
    model = VAEEndToEndFullyConnected(latent_dims=latent_dims, s_img=image_size[0], hdim=[hidden_dims, hidden_dims, hidden_dims])

    trainer = pl.Trainer(max_epochs=60, accelerator='gpu', devices=1)
    trainer.fit(model=model, datamodule=datamodule)

    for batch in datamodule.predict_dataloader():
        model.eval()
        with torch.no_grad():
            outputs = model(batch)

        rgb_gt = batch["model_input"]["rgb"]
        rgb_object_gt = batch["model_target"]["rgb_with_object"]
        mask_gt = batch["model_target"]["object_mask"]
        rgb_pred = outputs['rgb_with_object']
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
        break
    plt.savefig("results.png")
    plt.show()

