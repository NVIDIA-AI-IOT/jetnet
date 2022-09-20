import os

from typing import Optional, Tuple, Literal, Sequence

import numpy as np

import torch
import torch.nn.functional as F

from yolox.utils import postprocess

from jetnet.model import Model
from jetnet.config import Config
from jetnet.image import Image
from jetnet.classification import Classification
from jetnet.point import Point
from jetnet.polygon import Polygon
from jetnet.detection import (
    DetectionModel,
    Detection,
    DetectionSet
)
from jetnet.utils import download, make_parent_dir


__all__ = ["YOLOXModel"]


class PadResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        padded = (
            torch.ones(
                (3, self.size[0], self.size[1]),
                dtype=data.dtype,
                device=torch.device(data.device),
            )
            * 114
        )
        ar = min(padded.size(1) / data.size(1), padded.size(2) / data.size(2))
        new_size = (int(ar * data.size(1)), int(ar * data.size(2)))
        resized_data = F.interpolate(data[None, ...], size=new_size, mode="bilinear")[0]
        padded[:, 0 : new_size[0], 0 : new_size[1]] = resized_data
        return padded


class PilToTensor(object):
    def __call__(self, data: Image):
        data = np.array(data).transpose(2, 0, 1)[::-1]
        data = np.ascontiguousarray(data)
        tensor = torch.from_numpy(data)
        tensor = tensor.cuda().contiguous().float()
        return tensor


class YOLOXModel(DetectionModel):
    
    def __init__(self, module, device, input_size, labels, conf_thresh=0.3, nms_thresh=0.3,decoder=None):
        self._module = module.to(device).eval()
        self._device = device
        self._decoder = decoder
        self._labels = labels
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh
        self._pad_resize = PadResize(input_size[::-1])
        self._to_tensor = PilToTensor()
        self._input_size = input_size

    def get_labels(self) -> Sequence[str]:
        return self._labels

    def __call__(self, x: Image) -> Sequence[Detection]:
        with torch.no_grad():
            image = x
            width, height = image.width, image.height
            scale = 1.0 / min(self._input_size[1] / height, self._input_size[0] / width)
            data = self._to_tensor(image)
            data = self._pad_resize(data).to(self._device).float()[None, ...]
            outputs = self._module(data)
            if self._decoder is not None:
                outputs = self._decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, len(self._labels), self._conf_thresh, self._nms_thresh, class_agnostic=True
            )

            objects_tensor = outputs[0]

            if objects_tensor is None:
                return DetectionSet.construct(detections=[])

            num_objects = objects_tensor.size(0)
            objs = []
            for i in range(num_objects):
                o = objects_tensor[i].cpu()
                os = o[0:4] * scale
                obj = Detection.construct(
                    boundary=Polygon.construct(
                        points=[
                            Point.construct(x=int(os[0]), y=int(os[1])),
                            Point.construct(x=int(os[2]), y=int(os[1])),
                            Point.construct(x=int(os[2]), y=int(os[3])),
                            Point.construct(x=int(os[0]), y=int(os[3])),
                        ]
                    ),
                    classification=Classification.construct(
                        index=int(o[6]),
                        label=self._labels[int(o[6])],
                        score=float(o[4] * o[5]),
                    )
                )
                objs.append(obj)
            return DetectionSet.construct(detections=objs)

