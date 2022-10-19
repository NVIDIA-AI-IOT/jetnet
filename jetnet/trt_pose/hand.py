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


HAND_KEYPOINTS = [
    "wrist",
    "thumb_1", 
    "thumb_2", 
    "thumb_3", 
    "thumb_4", 
    "index_finger_1", 
    "index_finger_2", 
    "index_finger_3", 
    "index_finger_4", 
    "middle_finger_1", 
    "middle_finger_2", 
    "middle_finger_3", 
    "middle_finger_4", 
    "ring_finger_1", 
    "ring_finger_2", 
    "ring_finger_3", 
    "ring_finger_4", 
    "pinky_finger_1", 
    "pinky_finger_2", 
    "pinky_finger_3", 
    "pinky_finger_4"
]


HAND_SKELETON = [
    [0, 4], 
    [0, 8], 
    [0, 12], 
    [0, 16], 
    [0, 20], 
    [1, 2], 
    [2, 3], 
    [3, 4], 
    [5, 6], 
    [6, 7], 
    [7, 8], 
    [9, 10], 
    [10, 11], 
    [11, 12], 
    [13, 14], 
    [14, 15], 
    [15, 16], 
    [17, 18], 
    [18, 19], 
    [19, 20]
]

def _create_cfgs(_base_alias, _input_size, _model_name, _weights_url):
    cfg = TRTPose(
        model=_model_name,
        input_size=_input_size,
        keypoints=HAND_KEYPOINTS,
        skeleton=HAND_SKELETON,
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


RESNET18_HAND_224X224, RESNET18_HAND_224X224_TRT, RESNET18_HAND_224X224_TRT_FP16, RESNET18_HAND_224X224_TRT_INT8, RESNET18_HAND_224X224_TRT_INT8_DLA = \
    _create_cfgs(
        "resnet18_hand_224x224", 
        (224, 224), 
        "resnet18_baseline_att", 
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/hand_pose_resnet18_att_244_244.pth"
    )