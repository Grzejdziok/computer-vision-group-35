from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.synthetic_data_generator import SyntheticDataGenerator
from data.list_dataset import ListDataset


class SyntheticDataModule(pl.LightningDataModule):
    train_set: ListDataset
    val_set: ListDataset

    def __init__(self,
                 synthetic_data_generator: SyntheticDataGenerator,
                 num_train_samples: int,
                 num_val_samples: int,
                 batch_size: int,
                 ):
        super().__init__()
        self._synthetic_data_generator = synthetic_data_generator
        self._num_train_samples = num_train_samples
        self._num_val_samples = num_val_samples
        self._batch_size = batch_size

    def prepare_data(self):
        self.train_set = ListDataset(self._synthetic_data_generator.generate(self._num_train_samples))
        self.val_set = ListDataset(self._synthetic_data_generator.generate(self._num_val_samples))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size)
