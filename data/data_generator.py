from typing import List, Tuple
from data.training_sample import TrainingSample
from dataclasses import dataclass

from abc import ABC


class DataGenerator(ABC):

    def generate(self) -> List[TrainingSample]:
        raise NotImplementedError()


@dataclass
class DatasetStatistics:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    image_size: Tuple[int, int]
