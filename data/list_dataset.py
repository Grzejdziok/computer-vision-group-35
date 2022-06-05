from typing import List

import torch
import torch.utils.data
import torchvision.transforms.functional as TF

from data.training_sample import TrainingSample


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, training_samples: List[TrainingSample], augment: bool):
        self.training_samples = training_samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.training_samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        training_sample = self.training_samples[idx]
        if self.augment:
            return self._augment(training_sample)
        else:
            return training_sample

    @staticmethod
    def _augment(training_sample: TrainingSample) -> TrainingSample:
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._hflip(training_sample)
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._vflip(training_sample)
        return training_sample

    @staticmethod
    def _shallow_copy(training_sample: TrainingSample) -> TrainingSample:
        training_sample = training_sample.copy()
        training_sample["model_input"] = training_sample["model_input"].copy()
        training_sample["model_target"] = training_sample["model_target"].copy()
        return training_sample

    @staticmethod
    def _hflip(training_sample: TrainingSample) -> TrainingSample:
        training_sample = ListDataset._shallow_copy(training_sample)

        training_sample["model_input"]["rgb"] = TF.hflip(training_sample["model_input"]["rgb"]).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.hflip(training_sample["model_target"]["zoomed_object_rgb"]).float()
        training_sample["model_target"]["rgb_with_object"] = TF.hflip(training_sample["model_target"]["rgb_with_object"]).float()
        training_sample["model_target"]["zoomed_object_mask"] = TF.hflip(training_sample["model_target"]["zoomed_object_mask"]).bool()
        training_sample["model_target"]["object_mask"] = TF.hflip(training_sample["model_target"]["object_mask"]).bool()

        normalized_bounding_box_xyxy = training_sample["model_target"]["normalized_bounding_box_xyxy"]
        normalized_bounding_box_xyxy = torch.tensor([
            1. - normalized_bounding_box_xyxy[2],
            normalized_bounding_box_xyxy[1],
            1. - normalized_bounding_box_xyxy[0],
            normalized_bounding_box_xyxy[3],
        ], )
        training_sample["model_target"]["normalized_bounding_box_xyxy"] = normalized_bounding_box_xyxy.float()
        return training_sample

    @staticmethod
    def _vflip(training_sample: TrainingSample) -> TrainingSample:
        training_sample = ListDataset._shallow_copy(training_sample)

        training_sample["model_input"]["rgb"] = TF.vflip(training_sample["model_input"]["rgb"]).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.vflip(training_sample["model_target"]["zoomed_object_rgb"]).float()
        training_sample["model_target"]["rgb_with_object"] = TF.vflip(training_sample["model_target"]["rgb_with_object"]).float()
        training_sample["model_target"]["zoomed_object_mask"] = TF.vflip(training_sample["model_target"]["zoomed_object_mask"]).bool()
        training_sample["model_target"]["object_mask"] = TF.vflip(training_sample["model_target"]["object_mask"]).bool()

        normalized_bounding_box_xyxy = training_sample["model_target"]["normalized_bounding_box_xyxy"]
        normalized_bounding_box_xyxy = torch.tensor([
            normalized_bounding_box_xyxy[0],
            1. - normalized_bounding_box_xyxy[3],
            normalized_bounding_box_xyxy[2],
            1. - normalized_bounding_box_xyxy[1],
        ], )
        training_sample["model_target"]["normalized_bounding_box_xyxy"] = normalized_bounding_box_xyxy.float()
        return training_sample

    @staticmethod
    def _color_jitter(training_sample: TrainingSample) -> TrainingSample:
        training_sample = ListDataset._shallow_copy(training_sample)
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._brightness(training_sample)
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._hue(training_sample)
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._saturation(training_sample)
        if torch.rand(1) < 0.5:
            training_sample = ListDataset._contrast(training_sample)
        return training_sample

    @staticmethod
    def _brightness(training_sample: TrainingSample) -> TrainingSample:
        brightness_factor = 0.5 + torch.rand(1)
        training_sample["model_input"]["rgb"] = TF.adjust_brightness(training_sample["model_input"]["rgb"], brightness_factor).float()
        training_sample["model_target"]["rgb_with_object"] = TF.adjust_brightness(training_sample["model_target"]["rgb_with_object"], brightness_factor).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.adjust_brightness(training_sample["model_target"]["zoomed_object_rgb"], brightness_factor).float()
        return training_sample

    @staticmethod
    def _hue(training_sample: TrainingSample) -> TrainingSample:
        hue_factor = -0.5 + torch.rand(1)
        training_sample["model_input"]["rgb"] = TF.adjust_hue(training_sample["model_input"]["rgb"], hue_factor).float()
        training_sample["model_target"]["rgb_with_object"] = TF.adjust_hue(training_sample["model_target"]["rgb_with_object"], hue_factor).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.adjust_hue(training_sample["model_target"]["zoomed_object_rgb"], hue_factor).float()
        return training_sample

    @staticmethod
    def _saturation(training_sample: TrainingSample) -> TrainingSample:
        saturation_factor = 0.5 + torch.rand(1)
        training_sample["model_input"]["rgb"] = TF.adjust_saturation(training_sample["model_input"]["rgb"], saturation_factor).float()
        training_sample["model_target"]["rgb_with_object"] = TF.adjust_saturation(training_sample["model_target"]["rgb_with_object"], saturation_factor).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.adjust_saturation(training_sample["model_target"]["zoomed_object_rgb"], saturation_factor).float()
        return training_sample

    @staticmethod
    def _contrast(training_sample: TrainingSample) -> TrainingSample:
        contrast_factor = 0.5 + torch.rand(1)
        training_sample["model_input"]["rgb"] = TF.adjust_contrast(training_sample["model_input"]["rgb"], contrast_factor).float()
        training_sample["model_target"]["rgb_with_object"] = TF.adjust_contrast(training_sample["model_target"]["rgb_with_object"], contrast_factor).float()
        training_sample["model_target"]["zoomed_object_rgb"] = TF.adjust_contrast(training_sample["model_target"]["zoomed_object_rgb"], contrast_factor).float()
        return training_sample
