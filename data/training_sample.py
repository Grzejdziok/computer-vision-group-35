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
    zoomed_object_rgb: torch.FloatTensor
    zoomed_object_mask: torch.BoolTensor
    normalized_bounding_box_xyxy: torch.FloatTensor


class TrainingSample(TypedDict):
    model_input: ModelInput
    model_target: ModelTarget
    sample_id: str
