from typing import List
from data.training_sample import TrainingSample
import numpy as np

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class LossReporterCallback(pl.callbacks.Callback):

    def __init__(self):
        self._train_losses_per_epoch = []
        self._val_losses_per_epoch = []

        self._train_losses_per_batch = []
        self._val_losses_per_batch = []

        self._train_batch_sizes = []
        self._val_batch_sizes = []

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_losses_per_batch = []
        self._train_batch_sizes = []

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: torch.Tensor,
                           batch: TrainingSample,
                           batch_idx: int,
                           unused: int = 0,
                           ) -> None:
        batch_loss = outputs["loss"].item()
        self._train_losses_per_batch.append(batch_loss)
        self._train_batch_sizes.append(batch["model_input"]["rgb"].shape[0])

    @staticmethod
    def _calculate_total_loss(losses_per_batch: List[float], batch_sizes: List[float]) -> float:
        loss = float(np.sum(np.array(losses_per_batch) * np.array(batch_sizes)) / np.sum(batch_sizes))
        return loss

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_loss = self._calculate_total_loss(self._train_losses_per_batch, self._train_batch_sizes)
        self._train_losses_per_epoch.append(train_loss)
        self._train_losses_per_batch = []
        self._train_batch_sizes = []
        self.plot_and_save_losses("losses.png")
        plt.clf()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._val_losses_per_batch = []
        self._val_batch_sizes = []

    def on_validation_batch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: torch.Tensor,
                                batch: TrainingSample,
                                batch_idx: int,
                                unused: int = 0,
                                ) -> None:
        batch_loss = outputs.item()
        self._val_losses_per_batch.append(batch_loss)
        self._val_batch_sizes.append(batch["model_input"]["rgb"].shape[0])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_loss = self._calculate_total_loss(self._val_losses_per_batch, self._val_batch_sizes)
        self._val_losses_per_epoch.append(val_loss)
        self._val_losses_per_batch = []
        self._val_batch_sizes = []

    def get_train_losses(self) -> List[float]:
        return self._train_losses_per_epoch

    def get_val_losses(self) -> List[float]:
        return self._val_losses_per_epoch

    def get_epochs(self) -> List[int]:
        return list(range(1, len(self._train_losses_per_epoch)+1))

    def plot_and_save_losses(self, filename: str) -> None:
        np.save("train_losses.npy", self.get_train_losses())
        np.save("val_losses.npy", self.get_val_losses())
        np.save("epochs.npy", self.get_epochs())

        plt.plot(self.get_epochs(), self.get_train_losses(), label="train")
        plt.plot(self.get_epochs(), self.get_val_losses(), label="val")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim([0., 3.])
        plt.xlim([0, 6000])
        plt.title("Training and validation loss over training")
        plt.savefig(filename)
