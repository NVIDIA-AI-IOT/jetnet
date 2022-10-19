
from typing import Mapping, Sequence, Any, Literal, Optional
from pydantic import BaseModel

import os
import torch
import tempfile
from progressbar import ProgressBar, Bar, Timer, ETA

from torch2trt import torch2trt, trt, TRTModule
from torch2trt.flattener import Flattener
from torch2trt.dataset import FolderDataset

from jetnet.utils import make_parent_dir

Torch2trtCalibAlgo = Literal["legacy", "entropy", "entropy_2", "minmax"]
Torch2trtLogLevel = Literal["verbose", "error", "info", "warning"]
Torch2trtDeviceType = Literal["gpu", "dla"]


def trt_device_type_from_str(s: Torch2trtDeviceType):
    import tensorrt as trt
    if s.lower() == "gpu":
        return trt.DeviceType.GPU
    elif s.lower() == "dla":
        return trt.DeviceType.DLA
    else:
        raise ValueError(f"Unknown device type {s}.")

    
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
    int8_mode: bool = False
    fp16_mode: bool = False
    use_onnx: bool = False
    onnx_opset: int = 11
    log_level: Torch2trtLogLevel = "error"
    default_device_type: Torch2trtDeviceType = "gpu"
    engine_cache: Optional[str] = None
    calib_cache: Optional[str] = None
    num_calib: int = 1

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
    calib_dataset: Optional[BaseModel] = None

    def build(self):
        model = self.model.build()

        trt_modules = {}
        for name, config in self.engine_configs.items():
            if config.engine_cache is not None:
                engine_cache = os.path.expandvars(config.engine_cache)
                if os.path.exists(engine_cache):
                    module_trt = TRTModule()
                    module_trt.load_state_dict(torch.load(engine_cache))
                    trt_modules[name] = module_trt

        # generate cached data
        calib_datasets = {}

        if self.calib_dataset is not None \
            and any(cfg.int8_mode for cfg in self.engine_configs.values()) \
            and any(key not in trt_modules for key in self.engine_configs.keys()):

            calib_dataset = self.calib_dataset.build()
            calib_dataset_size = len(calib_dataset)
            for name, config in self.engine_configs.items():

                # skip cached models
                if name in trt_modules:
                    continue
                
                module = model.get_module(name)
                calib_cache = tempfile.mkdtemp() if config.calib_cache is None else config.calib_cache
                engine_calib_dataset = FolderDataset(calib_cache)
                calib_datasets[name] = engine_calib_dataset

                count = len(engine_calib_dataset)

                pbar = ProgressBar(
                    maxval=config.num_calib, 
                    widgets=[f"Generating INT8 calibration data for {name}[", Timer(), "] ", Bar(), " (", ETA(), ")"])
                pbar.start()

                # run inference to record calibration data
                while count < config.num_calib:
                    with engine_calib_dataset.record(module):
                        data = calib_dataset[count % calib_dataset_size]
                        model(data)
                    count += 1
                    pbar.update(count)
                pbar.finish()
        else:
            calib_datasets = {name: None for name in self.engine_configs.keys()}

        for name, config in self.engine_configs.items():
            if name in trt_modules:
                continue # skip cached
            module = model.get_module(name)
            inputs = config.build_input_tensors()
            module_trt = torch2trt(
                module,
                inputs,
                fp16_mode=config.fp16_mode,
                int8_mode=config.int8_mode,
                use_onnx=config.use_onnx,
                onnx_opset=config.onnx_opset,
                log_level=trt_log_level_from_str(config.log_level),
                min_shapes=config.get_min_shapes(),
                max_shapes=config.get_max_shapes(),
                opt_shapes=config.get_opt_shapes(),
                default_device_type=trt_device_type_from_str(config.default_device_type),
                int8_calib_dataset=calib_datasets[name]
            )
            trt_modules[name] = module_trt
            if config.engine_cache is not None:
                make_parent_dir(config.engine_cache)
                torch.save(module_trt.state_dict(), config.engine_cache)
        
        for name, module_trt in trt_modules.items():
            model.set_module(name, module_trt)

        return model
