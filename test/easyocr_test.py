import pytest
import os

from jetnet.image import read_image
from jetnet.textocr import TEXTOCR_TEST_IMAGES
from jetnet.easyocr import (
    EASYOCR_EN,
    EASYOCR_EN_TRT,
    EASYOCR_EN_TRT_FP16,
    EASYOCR_EN_TRT_INT8_FP16
)


def test_easyocr_en():

    model = EASYOCR_EN.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt():

    model = EASYOCR_EN_TRT.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt_fp16():

    model = EASYOCR_EN_TRT_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0


def test_easyocr_en_trt_int8_fp16():

    model = EASYOCR_EN_TRT_INT8_FP16.build()

    image = read_image("assets/text.jpg")

    output = model(image)

    assert len(output.detections) > 0

