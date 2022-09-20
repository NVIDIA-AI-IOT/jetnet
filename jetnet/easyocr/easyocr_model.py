from typing import Sequence

import torch

import cv2
import numpy as np

from easyocr import Reader

from jetnet.text_detection import (
    TextDetection,
    TextDetectionModel,
    TextDetectionSet
)
from jetnet.polygon import Polygon
from jetnet.point import Point
from jetnet.image import Image


__all__ = ["EasyOCR"]


class EasyOCRModel(TextDetectionModel):
    
    def __init__(self, reader: Reader):
        self._reader = reader

    @torch.no_grad()
    def __call__(self, x: Image) -> TextDetectionSet:
        image = x
        data = np.array(image)

        # RGB -> BGR
        if image.mode == "RGB":
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        raw_output = self._reader.readtext(data)

        detections = []

        for raw_value in raw_output:
            detection = TextDetection.construct(
                boundary=Polygon.construct(points=[
                    Point.construct(x=int(p[0]), y=int(p[1])) 
                    for p in raw_value[0]
                ]),
                text=raw_value[1],
                score=raw_value[2],
            )
            detections.append(detection)

        return TextDetectionSet.construct(detections=detections)