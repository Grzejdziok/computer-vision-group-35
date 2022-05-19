import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.list_dataset import ListDataset
import pytorch_lightning as pl
from data.data_generator import DataGenerator, DatasetStatistics
from data.training_sample import TrainingSample


class SingleItemGenerationDataModule(pl.LightningDataModule):
    train_set: ListDataset

    def __init__(self,
                 data_generator: DataGenerator,
                 batch_size: int,
                 ):
        super().__init__()
        self._data_generator = data_generator
        self._batch_size = batch_size
        self._dataset = None
        self._statistics = None  # lazy-loaded

    def prepare_data(self):
        generated_data = self._data_generator.generate()
        self._dataset = ListDataset(generated_data)

    def train_dataloader(self):
        assert self._dataset is not None
        return DataLoader(self._dataset, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        assert self._dataset is not None
        return DataLoader(self._dataset, batch_size=self._batch_size, shuffle=False)

    def test_dataloader(self):
        assert self._dataset is not None
        return DataLoader(self._dataset, batch_size=self._batch_size, shuffle=False)

    def predict_dataloader(self):
        assert self._dataset is not None
        return DataLoader(self._dataset, batch_size=self._batch_size, shuffle=True)

    @property
    def statistics(self) -> DatasetStatistics:
        if self._statistics is not None:
            return self._statistics

        image_size = None
        for batch in tqdm(self.train_dataloader(), desc="Asserting types and image sizes"):
            batch: TrainingSample
            rgb = batch["model_input"]["rgb"]
            rgb_with_object = batch["model_target"]["rgb_with_object"]
            object_mask = batch["model_target"]["object_mask"]
            assert isinstance(rgb, torch.FloatTensor)
            assert isinstance(rgb_with_object, torch.FloatTensor)

            image_size = image_size or (rgb.shape[2], rgb.shape[3])
            assert rgb.shape[2] == image_size[0] and rgb.shape[3] == image_size[1]
            assert object_mask.shape[1] == image_size[0] and object_mask.shape[2] == image_size[1]
            assert rgb_with_object.shape[2] == image_size[0] and rgb_with_object.shape[3] == image_size[1]

        dataset_mean = torch.zeros((3,))
        samples_included_in_mean = 0
        for batch in tqdm(self.train_dataloader(), desc="Calculating dataset mean"):
            rgb = batch["model_input"]["rgb"]
            batch_num_samples = rgb.shape[0]
            batch_mean = rgb.mean(dim=[0, 2, 3])
            dataset_mean = (dataset_mean * samples_included_in_mean + batch_num_samples * batch_mean) / (
                        samples_included_in_mean + batch_num_samples)
            samples_included_in_mean += batch_num_samples
        dataset_var = torch.zeros((3,))
        samples_included_in_var = 0
        for batch in tqdm(self.train_dataloader(), desc="Calculating dataset var"):
            rgb = batch["model_input"]["rgb"]
            batch_num_samples = rgb.shape[0]
            batch_var = ((rgb.permute((0, 2, 3, 1)) - dataset_mean) ** 2).mean(dim=[0, 1, 2])
            dataset_var = (dataset_var * samples_included_in_var + batch_num_samples * batch_var) / (
                        samples_included_in_var + batch_num_samples)
            samples_included_in_var += batch_num_samples
        self._statistics = DatasetStatistics(mean=tuple(dataset_mean), std=tuple(dataset_var ** (0.5)), image_size=image_size)
        return self._statistics
