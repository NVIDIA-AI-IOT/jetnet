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

from jetnet.mmocr.mmocr import MMOCR, MMOCRTRT
import jetnet.tensorrt
import jetnet.textocr


# TRT CONFIGS
MMOCR_DETECTOR_TRT_CFG = jetnet.tensorrt.Torch2trtConfig(
    use_onnx=True,
    int8_num_calib=128,
    min_shapes=[(1, 3, 240, 320)],
    opt_shapes=[(1, 3, 720, 1280)],
    max_shapes=[(1, 3, 1080, 1920)],
    int8_calib_algorithm="minmax"
)

MMOCR_RECOGNIZER_TRT_CFG = jetnet.tensorrt.Torch2trtConfig(
    use_onnx=True,
    int8_num_calib=128,
    min_shapes=[(1, 1, 16, 32)],
    opt_shapes=[(1, 1, 64, 320)],
    max_shapes=[(1, 1, 64, 1920)]
)

MMOCR_RECOGNIZER_TRT_CFG_3CH = jetnet.tensorrt.Torch2trtConfig(
    use_onnx=True,
    int8_num_calib=128,
    min_shapes=[(1, 3, 16, 32)],
    opt_shapes=[(1, 3, 64, 320)],
    max_shapes=[(1, 3, 64, 1920)]
)

# BASE MODEL CONFIGS


# DET TRT CFGS

MMOCR_DET_DB_R18_TRT_CFG = MMOCR_DETECTOR_TRT_CFG.copy(update={
    "engine_cache": f"data/mmocr/mmocr_det_db_r18_trt.pth",
    "int8_calib_cache": f"data/mmocr/mmocr_det_db_r18_calib"
})
MMOCR_DET_DB_R18_TRT_CFG_FP16 = MMOCR_DET_DB_R18_TRT_CFG.copy(update={
    "fp16_mode": True,
    "engine_cache": f"data/mmocr/mmocr_det_db_r18_trt_fp16.pth",
})
MMOCR_DET_DB_R18_TRT_CFG_INT8 = MMOCR_DET_DB_R18_TRT_CFG.copy(update={
    "int8_mode": True,
    "engine_cache": f"data/mmocr/mmocr_det_db_r18_trt_int8.pth",
})

# REC TRT CFGS

MMOCR_REC_CRNN_TRT_CFG = MMOCR_RECOGNIZER_TRT_CFG.copy(update={
    "engine_cache": f"data/mmocr/mmocr_rec_crnn_trt.pth",
    "int8_calib_cache": f"data/mmocr/mmocr_rec_crnn_trt_calib"
})
MMOCR_REC_CRNN_TRT_CFG_FP16 = MMOCR_REC_CRNN_TRT_CFG.copy(update={
    "fp16_mode": True,
    "engine_cache": f"data/mmocr/mmocr_rec_crnn_trt_fp16.pth",
})

MMOCR_REC_ROBUSTSCANNER_TRT_CFG = MMOCR_RECOGNIZER_TRT_CFG.copy(update={
    "engine_cache": f"data/mmocr/mmocr_rec_robustscanner_trt.pth",
    "int8_calib_cache": f"data/mmocr/mmocr_rec_robustscanner_trt_calib"
})
MMOCR_REC_ROBUSTSCANNER_TRT_CFG_FP16 = MMOCR_REC_ROBUSTSCANNER_TRT_CFG.copy(update={
    "fp16_mode": True,
    "engine_cache": f"data/mmocr/mmocr_rec_robustscanner_trt_fp16.pth",
})


# DBR18 - CRNN

MMOCR_DB_R18_CRNN = MMOCR(
    detector="DB_r18",
    recognizer="CRNN"
)

MMOCR_DB_R18_CRNN_TRT = MMOCRTRT(
    model=MMOCR_DB_R18_CRNN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG,
    recognizer_config=MMOCR_REC_CRNN_TRT_CFG
)

MMOCR_DB_R18_CRNN_TRT_FP16 = MMOCRTRT(
    model=MMOCR_DB_R18_CRNN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG_FP16,
    recognizer_config=MMOCR_REC_CRNN_TRT_CFG_FP16
)

MMOCR_DB_R18_CRNN_TRT_INT8_FP16 = MMOCRTRT(
    model=MMOCR_DB_R18_CRNN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG_INT8,
    recognizer_config=MMOCR_REC_CRNN_TRT_CFG_FP16
)


MMOCR_DB_R18_ROBUSTSCANNER = MMOCR(
    detector="DB_r18",
    recognizer="CRNN"
)

MMOCR_DB_R18_ROBUSTSCANNER_TRT = MMOCRTRT(
    model=MMOCR_DB_R18_ROBUSTSCANNER,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG,
    recognizer_config=MMOCR_REC_ROBUSTSCANNER_TRT_CFG
)

MMOCR_DB_R18_ROBUSTSCANNER_TRT_FP16 = MMOCRTRT(
    model=MMOCR_DB_R18_ROBUSTSCANNER,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG_FP16,
    recognizer_config=MMOCR_REC_ROBUSTSCANNER_TRT_CFG_FP16
)

MMOCR_DB_R18_ROBUSTSCANNER_TRT_INT8_FP16 = MMOCRTRT(
    model=MMOCR_DB_R18_ROBUSTSCANNER,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=MMOCR_DET_DB_R18_TRT_CFG_INT8,
    recognizer_config=MMOCR_REC_ROBUSTSCANNER_TRT_CFG_FP16
)

# TRT CONFIGS (FP16)