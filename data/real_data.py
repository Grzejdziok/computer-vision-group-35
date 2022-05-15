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
                 train_ratio: float,
                 batch_size: int,
                 resize: bool,
                 resize_dims: Tuple[int, int],
                 masks_dir: str,
                 image_dir: str,
                 datamodule_dir: str
                 ):
        super().__init__()
        self._real_data_generator = real_data_generator
        self._train_ratio = train_ratio
        self._batch_size = batch_size
        self._masks_dir = masks_dir
        self._image_dir = image_dir
        self._datamodule_dir = datamodule_dir
        self.resize = resize
        self.resize_dims = resize_dims

    def prepare_data(self):
        self.train_set = ListDataset(self._real_data_generator.generate("train", self._train_ratio, resize=self.resize, resize_dims=self.resize_dims, masks_dir=self._masks_dir, image_dir=self._image_dir, datamodule_dir=self._datamodule_dir))
        self.val_set = ListDataset(self._real_data_generator.generate("test", self._train_ratio, resize=self.resize, resize_dims=self.resize_dims, masks_dir=self._masks_dir, image_dir=self._image_dir, datamodule_dir=self._datamodule_dir))
        del self._real_data_generator #its very large

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=True)