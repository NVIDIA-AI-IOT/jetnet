import os
from argparse import Namespace
from typing import Literal


import numpy as np
import cv2
import torch

from mmocr.utils.ocr import MMOCR as MMOCR_original

from jetnet.config import Config
from jetnet.image import Image
from jetnet.text_detection import (
    TextDetection,
    TextDetectionModel,
    TextDetectionSet,
    TextDetectionModelConfig
)
from jetnet.point import Point
from jetnet.polygon import Polygon


__all__ = ["MMOCR"]


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

class MMOCRModel(TextDetectionModel):
    def __init__(self, mmocr: _MMOCR_wrapper):
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


class MMOCRModelConfig(TextDetectionModelConfig):

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

    def build(self) -> TextDetectionModel:
        _mmocr = _MMOCR_wrapper(
            det=self.detector,
            recog=self.recognizer,
            config_dir=os.path.join(os.environ["MMOCR_DIR"], "configs")
        )
        return MMOCRModel(_mmocr)