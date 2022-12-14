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

from .torchvision import TorchvisionModel, TorchvisionModelTRT
from jetnet.coco import COCO2017_VAL_IMAGES


def _create_cfgs(name):
    cfg = TorchvisionModel(name=name, pretrained=True)
    cfg_trt = TorchvisionModelTRT(
        model=cfg,
        engine_cache=f"data/torchvision/{name}_trt.pth"
    )
    cfg_trt_fp16 = TorchvisionModelTRT(
        model=cfg,
        fp16_mode=True,
        engine_cache=f"data/torchvision/{name}_trt_fp16.pth"
    )
    cfg_trt_int8 = TorchvisionModelTRT(
        model=cfg,
        int8_mode=True,
        engine_cache=f"data/torchvision/{name}_trt_int8.pth",
        int8_calib_dataset=COCO2017_VAL_IMAGES,
        int8_calib_cache=f"data/torchvision/{name}_calib",
        int8_num_calib=64
    )
    return cfg, cfg_trt, cfg_trt_fp16, cfg_trt_int8

# RESNET configs

RESNET18_IMAGENET, RESNET18_IMAGENET_TRT, RESNET18_IMAGENET_TRT_FP16, RESNET18_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet18")

RESNET34_IMAGENET, RESNET34_IMAGENET_TRT, RESNET34_IMAGENET_TRT_FP16, RESNET34_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet34")

RESNET50_IMAGENET, RESNET50_IMAGENET_TRT, RESNET50_IMAGENET_TRT_FP16, RESNET50_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet50")

RESNET101_IMAGENET, RESNET101_IMAGENET_TRT, RESNET101_IMAGENET_TRT_FP16, RESNET101_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet101")

RESNET152_IMAGENET, RESNET152_IMAGENET_TRT, RESNET152_IMAGENET_TRT_FP16, RESNET152_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet152")

DENSENET121_IMAGENET, DENSENET121_IMAGENET_TRT, DENSENET121_IMAGENET_TRT_FP16, DENSENET121_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet121")

DENSENET161_IMAGENET, DENSENET161_IMAGENET_TRT, DENSENET161_IMAGENET_TRT_FP16, DENSENET161_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet161")

DENSENET169_IMAGENET, DENSENET169_IMAGENET_TRT, DENSENET169_IMAGENET_TRT_FP16, DENSENET169_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet169")

DENSENET201_IMAGENET, DENSENET201_IMAGENET_TRT, DENSENET201_IMAGENET_TRT_FP16, DENSENET201_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet201")


MOBILENET_V2_IMAGENET, MOBILENET_V2_IMAGENET_TRT, MOBILENET_V2_IMAGENET_TRT_FP16, MOBILENET_V2_IMAGENET_TRT_INT8 = \
    _create_cfgs("mobilenet_v2")