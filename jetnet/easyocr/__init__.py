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

from jetnet.easyocr.easyocr import (
    EasyOCR,
    EasyOCRTRT,
    Torch2trtConfig,
    EASYOCR_TRT_DEFAULT_DETECTOR_CFG,
    EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG
)
import jetnet.textocr


EASYOCR_EN = EasyOCR(
    lang_list=["en"]
)

EASYOCR_EN_TRT = EasyOCRTRT(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "engine_cache": "data/easyocr/easyocr_en_det_trt.pth",
        "int8_calib_cache": "data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "engine_cache": "data/easyocr/easyocr_en_rec_trt.pth",
        "int8_calib_cache": "data/easyocr/easyocr_en_rec_calib"
    })
)

EASYOCR_EN_TRT_FP16 = EasyOCRTRT(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_det_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_rec_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_rec_calib"
    })
)

EASYOCR_EN_TRT_INT8_FP16 = EasyOCRTRT(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "int8_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_det_trt_int8.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_rec_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_rec_calib"
    })
)
