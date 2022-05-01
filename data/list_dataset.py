from typing import List

import torch.utils.data

from data.training_sample import TrainingSample


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, training_samples: List[TrainingSample]):
        self.training_samples = training_samples

    def __len__(self) -> int:
        return len(self.training_samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.training_samples[idx]
