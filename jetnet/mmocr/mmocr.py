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

import os
from argparse import Namespace
from typing import Literal


import numpy as np
import cv2
import torch

from mmocr.utils.ocr import MMOCR as MMOCR_original

from jetnet.image import Image
from jetnet.text_detection import (
    TextDetection,
    TextDetectionModel,
    TextDetectionSet
)
from jetnet.point import Point
from jetnet.polygon import Polygon
from pydantic import PrivateAttr
import tempfile
import os

from progressbar import ProgressBar, Bar, Timer, ETA

from typing import Literal, Optional, Tuple, Sequence, Optional

import numpy as np
import torch

from torch2trt import torch2trt, trt, TRTModule
from torch2trt.dataset import FolderDataset

import PIL.Image 
from jetnet.dataset import Dataset
from jetnet.image import Image, ImageDataset
from jetnet.text_detection import TextDetectionModel, TextDetectionSet
from jetnet.utils import make_parent_dir
from jetnet.tensorrt import trt_calib_algo_from_str, Int8CalibAlgo, trt_log_level_from_str, TrtLogLevel
from jetnet.tensorrt import Torch2trtConfig
from jetnet.config import Config


class _MMOCR_wrapper(MMOCR_original):
    def readtext_raw(
        self,
        img,
        output=None,
        details=False,
        export=None,
        export_format="json",
        batch_mode=False,
        recog_batch_size=0,
        det_batch_size=0,
        single_batch_size=0,
        imshow=False,
        print_result=False,
        merge=False,
        merge_xdist=20,
        **kwargs
    ):
        args = locals().copy()
        [args.pop(x, None) for x in ["kwargs", "self"]]
        args = Namespace(**args)

        # Input and output arguments processing
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model
            )
        return det_recog_result

class _MMOCR(TextDetectionModel):

    def __init__(self, mmocr):
        self._mmocr = mmocr
        
    @torch.no_grad()
    def __call__(self, x: Image) -> TextDetectionSet:

        image = x
        data = np.array(image)

        # RGB -> BGR
        if image.mode == "RGB":
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        raw_output = self._mmocr.readtext_raw(data)
        detections = []

        for raw_value in raw_output[0]["result"]:
            box = raw_value["box"]
            detection = TextDetection.construct(
                boundary=Polygon.construct(
                    points=[
                        Point.construct(x=int(box[2 * i]), y=int(box[2 * i + 1]))
                        for i in range(len(box) // 2)
                    ]
                ),
                text=raw_value["text"],
                score=float(raw_value["box_score"] * raw_value["text_score"]),
            )
            detections.append(detection)

        return TextDetectionSet.construct(detections=detections)


class MMOCR(Config[_MMOCR]):


    detector: Literal[
        "DB_r18",
        "DB_r50",
        "DBPP_r50",
        "DRRG",
        "FCE_IC15",
        "FCE_CTW_DCNv2",
        "MaskRCNN_CTW",
        "MaskRCNN_IC15",
        "MaskRCNN_IC17",
        "PANet_CTW",
        "PS_CTW",
        "PS_IC15",
        "TextSnake",
        "Tesseract",
    ]
    recognizer: Literal[
        "CRNN",
        "SAR",
        "SAR_CN",
        "NRTR_1/16-1/8",
        "NRTR_1/8-1/4",
        "RobustScanner",
        "SATRN",
        "SATRN_sm",
        "ABINet",
        "ABINet_Vision",
        "SEG",
        "CRNN_TPS",
        "Tesseract",
        "MASTER",
    ]

    def build(self):
        _mmocr = _MMOCR_wrapper(
            det=self.detector,
            recog=self.recognizer,
            config_dir=os.path.join(os.environ["MMOCR_DIR"], "configs")
        )
        return _MMOCR(_mmocr)


class MMOCRTRT(TextDetectionModel):
    model: MMOCR
    int8_calib_dataset: Optional[Config[ImageDataset]] = None
    detector_config: Optional[Torch2trtConfig] = None
    recognizer_config: Optional[Torch2trtConfig] = None
    
    def _build_trt(self, model, module, cfg: Torch2trtConfig):

        if cfg.engine_cache is not None and os.path.exists(cfg.engine_cache):
            module = TRTModule()
            module.load_state_dict(torch.load(cfg.engine_cache))
            return module
        
        if cfg.int8_mode:
            
            if cfg.int8_calib_cache is None:
                cfg.int8_calib_cache = tempfile.mkdtemp()

            calib_folder = FolderDataset(cfg.int8_calib_cache)
            
            if len(calib_folder) < cfg.int8_num_calib:

                assert self.int8_calib_dataset is not None

                int8_calib_dataset = self.int8_calib_dataset.build()
                i = len(calib_folder)

                pbar = ProgressBar(maxval=cfg.int8_num_calib, widgets=["Generating INT8 calibration data [", Timer(), "] ", Bar(), " (", ETA(), ")"])
                pbar.start()
                with calib_folder.record(module):
                    while len(calib_folder) < cfg.int8_num_calib:
                        x = int8_calib_dataset[i % len(int8_calib_dataset)]
                        model(x)
                        i += 1
                        pbar.update(i)
                pbar.finish()
        else:
            calib_folder = None
        
        module.cuda().eval()

        data = [torch.randn(cfg.opt_shapes[0]).cuda()]

        module_trt = torch2trt(
            module,
            data,
            fp16_mode=cfg.fp16_mode,
            int8_mode=cfg.int8_mode,
            max_workspace_size=cfg.max_workspace_size,
            int8_calib_dataset=calib_folder,
            int8_calib_algorithm=trt_calib_algo_from_str(cfg.int8_calib_algorithm),
            min_shapes=cfg.min_shapes,
            max_shapes=cfg.max_shapes,
            opt_shapes=cfg.opt_shapes,
            use_onnx=cfg.use_onnx,
            onnx_opset=cfg.onnx_opset,
            log_level=trt_log_level_from_str(cfg.log_level)
        )

        if cfg.engine_cache is not None:
            make_parent_dir(cfg.engine_cache)
            torch.save(module_trt.state_dict(), cfg.engine_cache)

        return module_trt

    def build(self):
        model = self.model.build()

        if self.detector_config is not None:
            det = model._mmocr.detect_model.backbone
            det_trt = self._build_trt(model, det, self.detector_config)

        if self.recognizer_config is not None:
            rec = model._mmocr.recog_model.backbone
            rec_trt = self._build_trt(model, rec, self.recognizer_config)

        if self.detector_config is not None:
            model._mmocr.detect_model.backbone = det_trt
        if self.recognizer_config is not None:
            model._mmocr.recog_model.backbone = rec_trt

        return model
