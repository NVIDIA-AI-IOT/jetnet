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


from typing import Literal, Optional, Sequence
from pydantic import BaseModel


Int8CalibAlgo = Literal["legacy", "entropy", "entropy_2", "minmax"]
TrtLogLevel = Literal["verbose", "error", "info", "warning"]


def trt_calib_algo_from_str(s: Int8CalibAlgo):
    import tensorrt as trt
    if s == "legacy":
        return trt.CalibrationAlgoType.LEGACY_CALIBRATION
    elif s == "entropy":
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    elif s == "entropy_2":
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    elif s == "minmax":
        return trt.CalibrationAlgoType.MINMAX_CALIBRATION
    else:
        raise ValueError(f"Unsupported calibration type {s}.")


def trt_log_level_from_str(s: TrtLogLevel):
    import tensorrt as trt
    if s == "verbose":
        return trt.Logger.VERBOSE
    elif s == "error":
        return trt.Logger.ERROR
    elif s == "info":
        return trt.Logger.INFO
    elif s == "warning":
        return trt.Logger.WARNING
    elif s == "internal_error":
        return trt.Logger.INTERNAL_ERROR
    else:
        raise ValueError("Unsupported log level type.")


class Torch2trtConfig(BaseModel):
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 26
    engine_cache: Optional[str] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 16
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"
    min_shapes: Optional[Sequence[Sequence[int]]] = None
    max_shapes: Optional[Sequence[Sequence[int]]] = None
    opt_shapes: Optional[Sequence[Sequence[int]]] = None
    use_onnx: bool = False
    onnx_opset: int = 11
    log_level: TrtLogLevel = "error"
