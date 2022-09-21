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


import pytest
import os

from jetnet.image import read_image
from jetnet.trt_pose import (
    RESNET18_BODY_224X224, 
    RESNET18_BODY_224X224_TRT,
    RESNET18_BODY_224X224_TRT_FP16,
    RESNET18_BODY_224X224_TRT_INT8
)
from jetnet.coco import COCO2017_VAL_IMAGES


def test_resnet18_body_224x224():

    model = RESNET18_BODY_224X224.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.poses) >= 1
    assert len(output.poses[0].keypoints) >= 1


def test_resnet18_body_224x224_trt():

    model = RESNET18_BODY_224X224_TRT.build()
    
    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.poses) >= 1
    assert len(output.poses[0].keypoints) >= 1


def test_resnet18_body_224x224_trt_fp16():

    model = RESNET18_BODY_224X224_TRT_FP16.build()
    
    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.poses) >= 1
    assert len(output.poses[0].keypoints) >= 1


def test_resnet18_body_224x224_trt_int8():

    model = RESNET18_BODY_224X224_TRT_INT8.build()
    
    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.poses) >= 1
    assert len(output.poses[0].keypoints) >= 1


