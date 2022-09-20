import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms


from typing import Tuple, Sequence, Literal, Optional

from jetnet.image import Image
from jetnet.classification import Classification, ClassificationModel
from jetnet.imagenet import IMAGENET_LABELS
from jetnet.image import Image
from jetnet.dataset import Dataset


__all__ = [
    "TorchvisionModel",
    "TorchvisionModelConfig"
]


class TorchvisionModel(ClassificationModel):

    def __init__(self, module, device, labels, input_size):
        self._device = torch.device(device)
        self._module = module.to(self._device).eval()
        self._normalize = torchvision.transforms.Normalize(
            [255.0 * 0.485, 255.0 * 0.456, 255.0 * 0.406],
            [255.0 * 0.229, 255.0 * 0.224, 255.0 * 0.225],
        ).to(self._device)
        self._labels = labels
        self._input_size = input_size

    def get_labels(self):
        return self._labels

    def __call__(self, x: Image) -> Classification:
        with torch.no_grad():
            tensor = (
                torch.from_numpy(np.array(x.pil()))
                .to(self._device)
                .permute(2, 0, 1)
                .float()
            )
            tensor = F.interpolate(tensor[None, ...], size=self._input_size)
            tensor = self._normalize(tensor[0])
            output = self._module(tensor[None, ...]).cpu()
            index = int(torch.argmax(output[0]))
            score = float(output[0, index])
            label = self.get_labels()[index]
            return Classification.construct(
                index=index,
                score=score,
                label=label
            )
