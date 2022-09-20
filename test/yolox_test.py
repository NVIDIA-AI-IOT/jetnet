import pytest
import os

from jetnet.image import read_image
from jetnet.coco import COCO_CLASSES, COCO2017_VAL_IMAGES
from jetnet.yolox import (
    YOLOX_TINY,
    YOLOX_TINY_TRT,
    YOLOX_TINY_TRT_FP16,
    YOLOX_TINY_TRT_INT8
)


def test_yolox_tiny():

    model = YOLOX_TINY.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.detections) > 0
    assert output.detections[0].classification.label == "person"


def test_yolox_tiny_trt():

    model = YOLOX_TINY_TRT.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.detections) > 0
    assert output.detections[0].classification.label == "person"


def test_yolox_tiny_trt_fp16():

    model = YOLOX_TINY_TRT_FP16.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.detections) > 0
    assert output.detections[0].classification.label == "person"


def test_yolox_tiny_trt_int8():

    model = YOLOX_TINY_TRT_INT8.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.detections) > 0
    assert output.detections[0].classification.label == "person"
