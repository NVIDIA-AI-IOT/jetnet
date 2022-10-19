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

import jetnet.coco
from jetnet.trt_pose.trt_pose import TRTPose
from jetnet.torch2trt import (
    Torch2trtModel,
    Torch2trtInputSpec,
    Torch2trtEngineConfig
)

BODY_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
]

BODY_SKELETON = [
    [15, 13], 
    [13, 11], 
    [16, 14], 
    [14, 12], 
    [11, 12], 
    [5, 7], 
    [6, 8], 
    [7, 9], 
    [8, 10], 
    [1, 2], 
    [0, 1], 
    [0, 2], 
    [1, 3], 
    [2, 4], 
    [3, 5], 
    [4, 6], 
    [17, 0], 
    [17, 5], 
    [17, 6], 
    [17, 11], 
    [17, 12]
]


def _create_cfgs(_base_alias, _input_size, _model_name, _weights_url):
    cfg = TRTPose(
        model=_model_name,
        input_size=_input_size,
        keypoints=BODY_KEYPOINTS,
        skeleton=BODY_SKELETON,
        weights_url=_weights_url,
        weights_path=f"data/trt_pose/{_base_alias}.pth"
    )
    cfg_trt = Torch2trtModel(
        model=cfg,
        engine_configs={
            "module": Torch2trtEngineConfig(
                inputs=[
                    Torch2trtInputSpec(
                        min_shape=[1, 3, _input_size[1], _input_size[0]],
                        max_shape=[1, 3, _input_size[1], _input_size[0]],
                        opt_shape=[1, 3, _input_size[1], _input_size[0]]
                    )
                ],
                engine_cache=f"data/trt_pose/{_base_alias}_trt.pth",
                use_onnx=True
            )
        }
    )
    cfg_trt_fp16 = cfg_trt.copy(deep=True)
    cfg_trt_fp16.engine_configs["module"].fp16_mode = True
    cfg_trt_fp16.engine_configs["module"].engine_cache = f"data/trt_pose/{_base_alias}_trt_fp16.pth"

    cfg_trt_int8 = cfg_trt.copy(deep=True)
    cfg_trt_int8.engine_configs["module"].int8_mode = True
    cfg_trt_int8.engine_configs["module"].engine_cache = f"data/trt_pose/{_base_alias}_trt_int8.pth"
    cfg_trt_int8.engine_configs["module"].num_calib = 512
    cfg_trt_int8.engine_configs["module"].calib_cache = f"data/trt_pose/{_base_alias}_calib"
    cfg_trt_int8.calib_dataset = jetnet.coco.COCO2017_VAL_IMAGES

    cfg_trt_int8_dla = cfg_trt_int8.copy(deep=True)
    cfg_trt_int8_dla.engine_configs["module"].engine_cache = f"data/trt_pose/{_base_alias}_trt_int8_dla.pth"
    cfg_trt_int8_dla.engine_configs["module"].default_device_type = "dla"

    return cfg, cfg_trt, cfg_trt_fp16, cfg_trt_int8, cfg_trt_int8_dla


RESNET18_BODY_224X224, RESNET18_BODY_224X224_TRT, RESNET18_BODY_224X224_TRT_FP16, RESNET18_BODY_224X224_TRT_INT8, RESNET18_BODY_224X224_TRT_INT8_DLA = \
    _create_cfgs(
        "resnet18_body_224x224", 
        (224, 224), 
        "resnet18_baseline_att",
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth"
    )


DENSENET121_BODY_256X256, DENSENET121_BODY_256X256_TRT, DENSENET121_BODY_256X256_TRT_FP16, DENSENET121_BODY_256X256_TRT_INT8, DENSENET121_BODY_256X256_TRT_INT8_DLA = \
    _create_cfgs(
        "densenet121_body_256x256", 
        (256, 256), 
        "densenet121_baseline_att",
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/densenet121_baseline_att_256x256_B_epoch_160.pth"
    )