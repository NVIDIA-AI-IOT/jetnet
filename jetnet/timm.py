from tty import CFLAG
from jetnet.config import Config
import timm
import torch
from typing import Sequence, Tuple, Literal
from jetnet.imagenet import IMAGENET_LABELS
from jetnet.torchvision.torchvision import _TorchvisionModel
from jetnet.torch2trt import (
    Torch2trtModel,
    Torch2trtInputSpec,
    Torch2trtEngineConfig
)
from jetnet.coco import COCO2017_VAL_IMAGES

class TimmModel(Config[_TorchvisionModel]):
    
    model: str
    input_size: Tuple[int, int] = (224, 224)
    pretrained: bool = False
    labels: Sequence[str] = IMAGENET_LABELS
    device: Literal["cpu", "cuda"] = "cuda"
    
    def build(self) -> _TorchvisionModel:
        device = torch.device(self.device)
        module = timm.create_model(self.model, pretrained=self.pretrained)
        module = module.to(device).eval()
        return _TorchvisionModel(module, device, self.input_size, self.labels)


def _make_pretrained_cfgs(name):

    cfg = TimmModel(model=name, pretrained=True)
    cfg_trt = Torch2trtModel(
        model=cfg,
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
                engine_cache=f"data/timm/{name}_trt.pth"
            )
        }
    )
    cfg_trt_fp16 = cfg_trt.copy(deep=True)
    cfg_trt_fp16.engine_configs["module"].fp16_mode = True
    cfg_trt_fp16.engine_configs["module"].engine_cache = f"data/timm/{name}_trt_fp16.pth"

    cfg_trt_int8 = cfg_trt.copy(deep=True)
    cfg_trt_int8.engine_configs["module"].int8_mode = True
    cfg_trt_int8.engine_configs["module"].engine_cache = f"data/timm/{name}_trt_int8.pth"
    cfg_trt_int8.engine_configs["module"].num_calib = 32
    cfg_trt_int8.calib_dataset = COCO2017_VAL_IMAGES

    cfg_trt_int8_dla = cfg_trt_int8.copy(deep=True)
    cfg_trt_int8_dla.engine_configs["module"].engine_cache = f"data/timm/{name}_trt_int8_dla.pth"
    cfg_trt_int8_dla.engine_configs["module"].default_device_type = "dla"

    return cfg, cfg_trt, cfg_trt_fp16, cfg_trt_int8, cfg_trt_int8_dla

RESNET18_IMAGENET, RESNET18_IMAGENET_TRT, RESNET18_IMAGENET_TRT_FP16, RESNET18_IMAGENET_TRT_INT8, \
    RESNET18_IMAGENET_TRT_INT8_DLA = _make_pretrained_cfgs("resnet18")