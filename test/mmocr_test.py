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
from jetnet.mmocr import (
    MMOCR_DB_R18_CRNN,
    MMOCR_DB_R18_CRNN_TRT_INT8_FP16,
    MMOCR_DB_R18_ROBUSTSCANNER,
    MMOCR_DB_R18_ROBUSTSCANNER_TRT_INT8_FP16
)


def test_mmocr_db_r18_crnn():

    model = MMOCR_DB_R18_CRNN.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_mmocr_db_r18_crnn_trt_int8_fp16():

    model = MMOCR_DB_R18_CRNN_TRT_INT8_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_mmocr_db_r18_robustscanner():

    model = MMOCR_DB_R18_ROBUSTSCANNER.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_mmocr_db_r18_robustscanner_trt_int8_fp16():

    model = MMOCR_DB_R18_ROBUSTSCANNER_TRT_INT8_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0
