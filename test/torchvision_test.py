import pytest

from jetnet.image import read_image
from jetnet.torchvision import (
    TorchvisionModelConfig,
    TorchvisionTRTModelConfig,
    RESNET18_IMAGENET,
    RESNET18_IMAGENET_TRT,
    RESNET18_IMAGENET_TRT_FP16,
    RESNET18_IMAGENET_TRT_INT8
)
from jetnet.coco import COCO2017_VAL_IMAGES


def test_resnet18():

    model = RESNET18_IMAGENET.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"


def test_resnet18_trt():

    model = RESNET18_IMAGENET_TRT.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"


def test_resnet18_trt_int8():

    model = RESNET18_IMAGENET_TRT_INT8.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207
    assert output.label == "golden retriever"