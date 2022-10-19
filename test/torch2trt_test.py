import pytest
from jetnet.torchvision import (
    RESNET18_IMAGENET
)
from jetnet.torch2trt import (
    Torch2trtModel,
    Torch2trtEngineConfig,
    Torch2trtInputSpec
)
from jetnet.image import read_image


def test_torch2trt_resnet18_onnx():

    config = Torch2trtModel(
        model=RESNET18_IMAGENET,
        engine_configs={
            "module": Torch2trtEngineConfig(
                inputs=[
                    Torch2trtInputSpec(
                        min_shape=[1, 3, 224, 224],
                        max_shape=[1, 3, 224, 224],
                        opt_shape=[1, 3, 224, 224]
                    )
                ],
                use_onnx=True
            )
        }
    )

    model = config.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207 # golden retriever


def test_torch2trt_resnet18_onnx_fp16():

    config = Torch2trtModel(
        model=RESNET18_IMAGENET,
        engine_configs={
            "module": Torch2trtEngineConfig(
                inputs=[
                    Torch2trtInputSpec(
                        min_shape=[1, 3, 224, 224],
                        max_shape=[1, 3, 224, 224],
                        opt_shape=[1, 3, 224, 224]
                    )
                ],
                use_onnx=True,
                fp16_mode=True
            )
        }
    )

    model = config.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207 # golden retriever



def test_torch2trt_resnet18_onnx_int8():

    config = Torch2trtModel(
        model=RESNET18_IMAGENET,
        engine_configs={
            "module": Torch2trtEngineConfig(
                inputs=[
                    Torch2trtInputSpec(
                        min_shape=[1, 3, 224, 224],
                        max_shape=[1, 3, 224, 224],
                        opt_shape=[1, 3, 224, 224]
                    )
                ],
                use_onnx=True,
                int8_mode=True
            )
        }
    )

    model = config.build()

    image = read_image("assets/dog.jpg")

    output = model(image)

    assert output.index == 207 # golden retriever