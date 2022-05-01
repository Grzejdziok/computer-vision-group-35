import pytorch_lightning as pl

from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.synthetic_data import SyntheticDataModule

from models.oracle import OracleModel
from models.VAE import LitVAE
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    num_train_samples = 1000
    num_val_samples = 100   
    image_size = (10, 10)
    square_size = 5
    batch_size = 10

    synthetic_data_generator = GaussianNoiseWithSquareSyntheticDataGenerator(image_size=image_size, square_size=square_size)
    datamodule = SyntheticDataModule(
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        synthetic_data_generator=synthetic_data_generator,
        batch_size=batch_size,
    )
    model = LitVAE(latent_dims = 2, s_img=image_size[0], hdim = [100, 50], device=device) #values from DL assignment

    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1)
    trainer.fit(model=model, datamodule=datamodule)

    image_index = np.random.randint(10)
    for batch, outputs in enumerate(zip(trainer.predict(model, datamodule),datamodule.test_dataloader())):
        rgb_gt = outputs[1]["model_input"]["rgb"]
        rgb_object_gt = outputs[1]["model_target"]["rgb_with_object"]
        mask_gt = outputs[1]["model_target"]["object_mask"]
        rgb_pred = outputs[0]['rgb_with_object']
        mask_pred = outputs[0]['soft_object_mask']
        if batch==0:
            fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=False)
            ax1.imshow(rgb_gt[image_index])
            ax1.set_title("RGB - input")
            ax2.imshow(rgb_object_gt[image_index])
            ax2.set_title("RGB - target")
            ax3.imshow(mask_gt[image_index])
            ax3.set_title("Mask - target")
            ax4.imshow(rgb_pred[image_index])
            ax4.set_title("RGB - predicted")
            ax5.imshow(mask_pred[image_index])
            ax5.set_title("Mask - predicted")
    plt.show()
