# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from typing import Optional, Tuple, Literal, Sequence

import numpy as np

import torch
import torch.nn.functional as F

from yolox.utils import postprocess

from jetnet.image import Image, ImageDataset
from jetnet.classification import Classification
from jetnet.point import Point
from jetnet.polygon import Polygon
from jetnet.detection import (
    DetectionModel,
    Detection,
    DetectionSet
)
from jetnet.utils import download, make_parent_dir

import torch
import os
from jetnet.utils import download, make_parent_dir
from typing import Literal, Tuple, Sequence, Optional
from jetnet.detection import DetectionModel
from jetnet.coco import COCO_CLASSES
import tempfile
import os

import torch
from torch2trt import torch2trt, TRTModule
from torch2trt.dataset import FolderDataset
import PIL.Image
import numpy as np
from pydantic import PrivateAttr
from progressbar import Timer, ETA, Bar, ProgressBar

from typing import Optional

from jetnet.utils import make_parent_dir
from jetnet.dataset import Dataset
from jetnet.image import Image
from jetnet.detection import DetectionModel
from jetnet.tensorrt import Int8CalibAlgo, trt_calib_algo_from_str



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


class YOLOX(DetectionModel):

    exp: Literal["yolox_l", "yolox_m", "yolox_nano", "yolox_s", "yolox_tiny", "yolox_x"]
    input_size: Tuple[int, int]
    labels: Sequence[str]
    conf_thresh: Optional[float] = 0.3
    nms_thresh: Optional[float] = 0.3
    device: Literal["cpu", "cuda"] = "cuda"
    weights_path: Optional[str] = None
    weights_url: Optional[str] = None

    _module = PrivateAttr()
    _device = PrivateAttr()
    _decoder = PrivateAttr()
    _pad_resize = PrivateAttr()
    _to_tensor = PrivateAttr()


    def _get_weights_url(self):
        if self.exp == "yolox_l":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
        elif self.exp == "yolox_m":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"
        elif self.exp == "yolox_s":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
        elif self.exp == "yolox_tiny":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
        elif self.exp == "yolox_nano":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
        elif self.exp == "yolox_x":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"
        else:
            raise KeyError("Weights not found.")

    def get_exp(self):
        if self.exp == "yolox_l":
            from .exps.yolox_l import Exp

            return Exp()
        elif self.exp == "yolox_m":
            from .exps.yolox_m import Exp

            return Exp()
        elif self.exp == "yolox_s":
            from .exps.yolox_s import Exp

            return Exp()
        elif self.exp == "yolox_tiny":
            from .exps.yolox_tiny import Exp

            return Exp()
        elif self.exp == "yolox_nano":
            from .exps.yolox_nano import Exp

            return Exp()
        elif self.exp == "yolox_x":
            from .exps.yolox_x import Exp

            return Exp()
        else:
            raise KeyError("Experiment not found.")

    def init(self):

        device = torch.device(self.device)

        exp = self.get_exp()
        exp.test_conf = self.conf_thresh
        exp.nmsthre = self.nms_thresh
        exp.test_size = self.input_size[::-1] # YOLOX uses (h, w), we use (w, h)

        module = exp.get_model()

        if self.weights_path is not None:
            if not os.path.exists(self.weights_path):
                make_parent_dir(self.weights_path)
                if self.weights_url is None:
                    self.weights_url = self._get_weights_url()
                download(self.weights_url, self.weights_path)
            module.load_state_dict(torch.load(self.weights_path)["model"])

        module = module.to(device).eval()

        self._pad_resize = PadResize(self.input_size[::-1])
        self._to_tensor = PilToTensor()
        self._module = module
        self._device = device
        self._decoder = None

    def get_labels(self) -> Sequence[str]:
        return self.labels

    def __call__(self, x: Image) -> Sequence[Detection]:
        with torch.no_grad():
            image = x
            width, height = image.width, image.height
            scale = 1.0 / min(self.input_size[1] / height, self.input_size[0] / width)
            data = self._to_tensor(image)
            data = self._pad_resize(data).to(self._device).float()[None, ...]
            outputs = self._module(data)
            if self._decoder is not None:
                outputs = self._decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, len(self.labels), self.conf_thresh, self.nms_thresh, class_agnostic=True
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
                        label=self.labels[int(o[6])],
                        score=float(o[4] * o[5]),
                    )
                )
                objs.append(obj)
            return DetectionSet.construct(detections=objs)


class YOLOXTRT(DetectionModel):

    model: YOLOX
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 25
    engine_cache: Optional[str] = None
    int8_calib_dataset: Optional[ImageDataset] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 512
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"

    def init(self):

        model = self.model.build()
        model._module.head.decode_in_inference = False

        if self.engine_cache is not None and os.path.exists(self.engine_cache):
            module_trt = TRTModule()
            module_trt.load_state_dict(torch.load(self.engine_cache))
        else:
            if self.int8_mode:
                
                if self.int8_calib_cache is None:
                    self.int8_calib_cache = tempfile.mkdtemp()

                calib_folder = FolderDataset(self.int8_calib_cache)
                
                if len(calib_folder) < self.int8_num_calib:

                    assert self.int8_calib_dataset is not None

                    int8_calib_dataset = self.int8_calib_dataset.build()
                    i = len(calib_folder)

                    pbar = ProgressBar(maxval=self.int8_num_calib, widgets=["Generating INT8 calibration data [", Timer(), "] ", Bar(), " (", ETA(), ")"])
                    pbar.start()
                    with calib_folder.record(model._module):
                        while len(calib_folder) < self.int8_num_calib:
                            x = int8_calib_dataset[i % len(int8_calib_dataset)]
                            model(x)
                            i += 1
                            pbar.update(i)
                    pbar.finish()
            else:
                calib_folder = None

            data = torch.randn((1, 3) + self.model.input_size[::-1]).to(model._device)

            module = model._module.to(model._device).eval()
            

            module_trt = torch2trt(
                module,
                [data],
                fp16_mode=self.fp16_mode,
                int8_mode=self.int8_mode,
                max_workspace_size=self.max_workspace_size,
                int8_calib_dataset=calib_folder,
                int8_calib_algorithm=trt_calib_algo_from_str(self.int8_calib_algorithm)
            )

            if self.engine_cache is not None:
                make_parent_dir(self.engine_cache)
                torch.save(module_trt.state_dict(), self.engine_cache)


        model._decoder = model._module.head.decode_outputs
        # run model once to do some configuration
        dummy_image = PIL.Image.fromarray(np.zeros(self.model.input_size[::-1] + (3,), dtype=np.uint8))
        model(dummy_image)
        model._module = module_trt

        self.model = model
        return self

    def __call__(self, x: Image) -> DetectionSet:
        return self.model(x)
    
    def get_labels(self):
        return self.model.get_labels()
        