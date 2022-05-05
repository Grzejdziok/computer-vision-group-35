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
                 num_train_samples: int,
                 num_val_samples: int,
                 batch_size: int,
                 resize: bool,
                 resize_dims: Tuple[int, int]
                 ):
        super().__init__()
        self._real_data_generator = real_data_generator
        self._num_train_samples = num_train_samples
        self._num_val_samples = num_val_samples
        self._batch_size = batch_size
        self.resize = resize
        self.resize_dims = resize_dims


    def prepare_data(self):
        self.train_set = ListDataset(self._real_data_generator.generate(self._num_train_samples, resize=self.resize, resize_dims=self.resize_dims))
        self.val_set = ListDataset(self._real_data_generator.generate(self._num_val_samples, resize=self.resize, resize_dims=self.resize_dims))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False)


