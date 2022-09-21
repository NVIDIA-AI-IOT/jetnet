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
from jetnet.textocr import TEXTOCR_TEST_IMAGES
from jetnet.easyocr import (
    EASYOCR_EN,
    EASYOCR_EN_TRT,
    EASYOCR_EN_TRT_FP16,
    EASYOCR_EN_TRT_INT8_FP16
)


def test_easyocr_en():

    model = EASYOCR_EN.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt():

    model = EASYOCR_EN_TRT.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt_fp16():

    model = EASYOCR_EN_TRT_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt_int8_fp16():

    model = EASYOCR_EN_TRT_INT8_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0

