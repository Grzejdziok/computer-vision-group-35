import torch

from typing import TypedDict


class ModelInput(TypedDict):
    rgb: torch.FloatTensor


class ModelOutput(TypedDict):
    rgb_with_object: torch.FloatTensor
    soft_object_mask: torch.FloatTensor


class ModelTarget(TypedDict):
    rgb_with_object: torch.FloatTensor
    object_mask: torch.BoolTensor
    object_rgb: torch.FloatTensor


class TrainingSample(TypedDict):
    model_input: ModelInput
    model_target: ModelTarget
