from .torchvision_model_config import TorchvisionModelConfig
from .torchvision_trt_model_config import TorchvisionTRTModelConfig
from jetnet.coco import COCO2017_VAL_IMAGES


def _create_cfgs(name):
    cfg = TorchvisionModelConfig(name=name, pretrained=True)
    cfg_trt = TorchvisionTRTModelConfig(
        model=cfg,
        engine_cache=f"data/torchvision/{name}_trt.pth"
    )
    cfg_trt_fp16 = TorchvisionTRTModelConfig(
        model=cfg,
        fp16_mode=True,
        engine_cache=f"data/torchvision/{name}_trt_fp16.pth"
    )
    cfg_trt_int8 = TorchvisionTRTModelConfig(
        model=cfg,
        int8_mode=True,
        engine_cache=f"data/torchvision/{name}_trt_int8.pth",
        int8_calib_dataset=COCO2017_VAL_IMAGES,
        int8_calib_cache=f"data/torchvision/{name}_calib",
        int8_num_calib=64
    )
    return cfg, cfg_trt, cfg_trt_fp16, cfg_trt_int8

# RESNET configs

RESNET18_IMAGENET, RESNET18_IMAGENET_TRT, RESNET18_IMAGENET_TRT_FP16, RESNET18_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet18")

RESNET34_IMAGENET, RESNET34_IMAGENET_TRT, RESNET34_IMAGENET_TRT_FP16, RESNET34_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet34")

RESNET50_IMAGENET, RESNET50_IMAGENET_TRT, RESNET50_IMAGENET_TRT_FP16, RESNET50_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet50")

RESNET101_IMAGENET, RESNET101_IMAGENET_TRT, RESNET101_IMAGENET_TRT_FP16, RESNET101_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet101")

RESNET152_IMAGENET, RESNET152_IMAGENET_TRT, RESNET152_IMAGENET_TRT_FP16, RESNET152_IMAGENET_TRT_INT8 = \
    _create_cfgs("resnet152")

DENSENET121_IMAGENET, DENSENET121_IMAGENET_TRT, DENSENET121_IMAGENET_TRT_FP16, DENSENET121_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet121")

DENSENET161_IMAGENET, DENSENET161_IMAGENET_TRT, DENSENET161_IMAGENET_TRT_FP16, DENSENET161_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet161")

DENSENET169_IMAGENET, DENSENET169_IMAGENET_TRT, DENSENET169_IMAGENET_TRT_FP16, DENSENET169_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet169")

DENSENET201_IMAGENET, DENSENET201_IMAGENET_TRT, DENSENET201_IMAGENET_TRT_FP16, DENSENET201_IMAGENET_TRT_INT8 = \
    _create_cfgs("densenet201")


MOBILENET_V2_IMAGENET, MOBILENET_V2_IMAGENET_TRT, MOBILENET_V2_IMAGENET_TRT_FP16, MOBILENET_V2_IMAGENET_TRT_INT8 = \
    _create_cfgs("mobilenet_v2")