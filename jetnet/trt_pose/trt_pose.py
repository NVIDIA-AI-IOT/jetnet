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

from jetnet.utils import make_parent_dir, download
from jetnet.pose import PoseModel, PoseSet, Keypoint, Pose
from jetnet.image import Image, ImageDataset
from jetnet.tensorrt import trt_calib_algo_from_str, Int8CalibAlgo

from typing import Literal, Optional, Sequence, Tuple, Literal, Callable
from pydantic import PrivateAttr
import os
import numpy as np
import tempfile
from progressbar import Timer, Bar, ETA, ProgressBar

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from torch2trt import torch2trt, TRTModule
from torch2trt.dataset import FolderDataset

from trt_pose.coco import coco_category_to_topology
from trt_pose.parse_objects import ParseObjects
from trt_pose.models import MODELS



class TRTPose(PoseModel):

    model: Literal[
        "resnet18_baseline_att",
        "densenet121_baseline_att"
    ]
    keypoints: Sequence[str]
    skeleton: Sequence[Tuple[int, int]]
    input_size: Tuple[int, int]
    weights_url: Optional[str] = None
    weights_path: Optional[str] = None
    device: Literal["cpu", "cuda"] = "cuda"

    _module = PrivateAttr()
    _device = PrivateAttr()
    _topology = PrivateAttr()
    _parse_objects = PrivateAttr()
    _normalize = PrivateAttr()
    
    def init(self):

        module = MODELS[self.model](len(self.keypoints), 2 * len(self.skeleton))

        if self.weights_path is not None:
            if not os.path.exists(self.weights_path):
                make_parent_dir(self.weights_path)
                download(self.weights_url, self.weights_path)
            module.load_state_dict(torch.load(self.weights_path))

        self._device = torch.device(self.device)
        self._module = module.to(self._device).eval()

        coco_skeleton = [[a+1, b+1] for a, b in self.skeleton]
        
        self._topology = coco_category_to_topology(
            {"keypoints": self.keypoints, "skeleton": coco_skeleton}
        )
        self._parse_objects = ParseObjects(self._topology)
        self._normalize = Normalize(
            [255.0 * 0.485, 255.0 * 0.456, 255.0 * 0.406],
            [255.0 * 0.229, 255.0 * 0.224, 255.0 * 0.225],
        ).to(self._device)

    def get_keypoints(self) -> Sequence[str]:
        return self.keypoints

    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        return self.skeleton

    def __call__(self, x: Image) -> PoseSet:
        with torch.no_grad():
            image = x
            width, height = image.width, image.height
            data = (
                torch.from_numpy(np.asarray(image))
                .to(self._device)
                .permute(2, 0, 1)
                .float()
            )[None, ...]

            data = F.interpolate(data, size=self.input_size[::-1])

            data = self._normalize(data[0])
            cmap, paf = self._module(data[None, ...])
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

            counts, objects, peaks = self._parse_objects(cmap, paf)
            count = int(counts[0])

            poses = []

            for i in range(count):

                keypoints = []
                obj = objects[0][i]
                c = obj.shape[0]

                for j in range(c):
                    k = int(obj[j])
                    if k >= 0:
                        peak = peaks[0][j][k]
                        kp = Keypoint.construct(
                            x = round(float(peak[1]) * width),
                            y = round(float(peak[0]) * height),
                            index = j,
                            label=self.get_keypoints()[j]
                        )
                        keypoints.append(kp)

                pose = Pose.construct(
                    keypoints=keypoints
                )

                poses.append(pose)

            return PoseSet.construct(poses=poses)


class TRTPoseTRT(PoseModel):

    model: TRTPose
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 25
    engine_cache: Optional[str] = None
    int8_calib_dataset: Optional[ImageDataset] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 1
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"

    def init(self):
        model = self.model.build()

        if self.engine_cache is not None and os.path.exists(self.engine_cache):
            module = TRTModule()
            module.load_state_dict(torch.load(self.engine_cache))
            model._module = module
            self.model = model
            return self
        
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

        model._module = module_trt

        self.model = model
        return self
    
    def get_keypoints(self) -> Sequence[str]:
        return self.model.get_keypoints()

    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        return self.model.get_skeleton()

    def __call__(self, x: Image) -> PoseSet:
        return self.model(x)

