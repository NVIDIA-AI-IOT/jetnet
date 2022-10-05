
from typing import Mapping, Sequence, Any, Literal
from pydantic import BaseModel

import torch

from torch2trt import torch2trt, trt, TRTModule
from torch2trt.flattener import Flattener


Torch2trtCalibAlgo = Literal["legacy", "entropy", "entropy_2", "minmax"]
Torch2trtLogLevel = Literal["verbose", "error", "info", "warning"]


def trt_calib_algo_from_str(s: Torch2trtCalibAlgo):
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


def trt_log_level_from_str(s: Torch2trtLogLevel):
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


class Torch2trtInputSpec(BaseModel):
    min_shape: Sequence[int]
    max_shape: Sequence[int]
    opt_shape: Sequence[int]


class Torch2trtEngineConfig(BaseModel):
    inputs: Any
    fp16_mode: bool = False
    use_onnx: bool = False
    onnx_opset: int = 11
    log_level: Torch2trtLogLevel = "error"

    def get_input_flattener(self):
        return Flattener.from_value(self.inputs, condition=lambda x: isinstance(x, Torch2trtInputSpec))

    def build_input_tensors(self):
        flattener = self.get_input_flattener()
        specs_flat = flattener.flatten(self.inputs)
        tensors_flat = [torch.randn(spec.opt_shape).cuda() for spec in specs_flat]
        return flattener.unflatten(tensors_flat)

    def get_min_shapes(self):
        flattener = self.get_input_flattener()
        specs_flat = flattener.flatten(self.inputs)
        shapes_flat = [spec.min_shape for spec in specs_flat]
        return flattener.unflatten(shapes_flat)

    def get_max_shapes(self):
        flattener = self.get_input_flattener()
        specs_flat = flattener.flatten(self.inputs)
        shapes_flat = [spec.max_shape for spec in specs_flat]
        return flattener.unflatten(shapes_flat)

    def get_opt_shapes(self):
        flattener = self.get_input_flattener()
        specs_flat = flattener.flatten(self.inputs)
        shapes_flat = [spec.opt_shape for spec in specs_flat]
        return flattener.unflatten(shapes_flat)


class Torch2trtModel(BaseModel):
    model: BaseModel
    engine_configs: Mapping[str, Torch2trtEngineConfig]

    def build(self):
        model = self.model.build()

        trt_modules = {}
        for name, config in self.engine_configs.items():
            module = model.get_module(name)
            inputs = config.build_input_tensors()
            module_trt = torch2trt(
                module,
                inputs,
                fp16_mode=config.fp16_mode,
                use_onnx=config.use_onnx,
                onnx_opset=config.onnx_opset,
                log_level=trt_log_level_from_str(config.log_level),
                min_shapes=config.get_min_shapes(),
                max_shapes=config.get_max_shapes(),
                opt_shapes=config.get_opt_shapes()
            )
            trt_modules[name] = module_trt
        
        for name, module_trt in trt_modules.items():
            model.set_module(name, module_trt)

        return model
