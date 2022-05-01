import pytorch_lightning as pl

from data.synthetic_data_generator import GaussianNoiseWithSquareSyntheticDataGenerator
from data.synthetic_data import SyntheticDataModule

from models.oracle import OracleModel


if __name__ == "__main__":

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
    model = OracleModel()

    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1)
    trainer.fit(model=model, datamodule=datamodule)
    predictions_list = trainer.predict(datamodule=datamodule, return_predictions=True)  # this is a list of predictions for each batch in predict_dataloader
