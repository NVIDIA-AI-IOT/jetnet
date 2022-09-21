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

from jetnet.image import read_image
from jetnet.torchvision import (
    RESNET18_IMAGENET,
    RESNET18_IMAGENET_TRT,
    RESNET18_IMAGENET_TRT_FP16,
    RESNET18_IMAGENET_TRT_INT8
)
from jetnet.coco import COCO2017_VAL_IMAGES


def test_resnet18():

    model = RESNET18_IMAGENET.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"


def test_resnet18_trt():

    model = RESNET18_IMAGENET_TRT.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"


def test_resnet18_trt_int8():

    model = RESNET18_IMAGENET_TRT_INT8.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"