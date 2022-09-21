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

from jetnet.yolox.yolox import YOLOX, YOLOXTRT
import jetnet.coco


def _create_configs(_exp, _input_size):
    _cfg = YOLOX(
        exp=_exp,
        input_size=_input_size,
        labels=jetnet.coco.COCO_CLASSES,
        conf_thresh=0.3,
        nms_thresh=0.3,
        weights_url=f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{_exp}.pth",
        weights_path=f"data/yolox/{_exp}.pth"
    )

    _cfg_trt = YOLOXTRT(
        model=_cfg, 
        engine_cache=f"data/yolox/{_exp}_trt.pth", 
        int8_calib_cache=f"data/yolox/{_exp}_calib",
        int8_calib_dataset=jetnet.coco.COCO2017_VAL_IMAGES
    )
    _cfg_trt_fp16 = _cfg_trt.copy(update={"fp16_mode": True, f"engine_cache": f"data/yolox/{_exp}_trt_fp16.pth"})
    _cfg_trt_int8 = _cfg_trt.copy(update={"int8_mode": True, f"engine_cache": f"data/yolox/{_exp}_trt_int8.pth"})

    return (_cfg, _cfg_trt, _cfg_trt_fp16, _cfg_trt_int8)


YOLOX_L, YOLOX_L_TRT, YOLOX_L_TRT_FP16, YOLOX_L_TRT_INT8 = _create_configs("yolox_l", (640, 640))
YOLOX_M, YOLOX_M_TRT, YOLOX_M_TRT_FP16, YOLOX_M_TRT_INT8 = _create_configs("yolox_m", (640, 640))
YOLOX_S, YOLOX_S_TRT, YOLOX_S_TRT_FP16, YOLOX_S_TRT_INT8 = _create_configs("yolox_s", (640, 640))
YOLOX_X, YOLOX_X_TRT, YOLOX_X_TRT_FP16, YOLOX_X_TRT_INT8 = _create_configs("yolox_x", (640, 640))
YOLOX_TINY, YOLOX_TINY_TRT, YOLOX_TINY_TRT_FP16, YOLOX_TINY_TRT_INT8 = _create_configs("yolox_tiny", (416, 416))
YOLOX_NANO, YOLOX_NANO_TRT, YOLOX_NANO_TRT_FP16, YOLOX_NANO_TRT_INT8 = _create_configs("yolox_nano", (416, 416))