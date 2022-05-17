from torch.utils.data import DataLoader
from data.list_dataset import ListDataset
import pytorch_lightning as pl
from data.real_data_generator import RealDataGenerator
from typing import Tuple, List


class RealDataModule(pl.LightningDataModule):
    train_set: ListDataset
    val_set: ListDataset

    def __init__(self,
                 real_data_generator: RealDataGenerator,
                 batch_size: int,
                 resize: bool,
                 resize_dims: Tuple[int, int],
                 dataset_dir: str,
                 datamodule_dir: str
                 ):
        super().__init__()
        self._real_data_generator = real_data_generator
        self._batch_size = batch_size
        self._dataset_dir = dataset_dir
        self._datamodule_dir = datamodule_dir
        self.resize = resize
        self.resize_dims = resize_dims

    def prepare_data(self):
        generated_data = self._real_data_generator.generate(
            resize=self.resize,
            resize_dims=self.resize_dims,
            dataset_dir=self._dataset_dir,
            datamodule_dir=self._datamodule_dir,
        )
        self.train_set = ListDataset(generated_data)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=True)