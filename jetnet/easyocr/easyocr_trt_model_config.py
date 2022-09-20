from torch2trt import torch2trt, trt, TRTModule
from torch2trt.dataset import FolderDataset

import tempfile
import os

from progressbar import ProgressBar, Bar, Timer, ETA

from typing import Literal, Optional, Tuple, Sequence, Optional

import numpy as np
import torch

from torch2trt import torch2trt, trt, TRTModule
from torch2trt.dataset import FolderDataset

import PIL.Image 
from jetnet.config import Config
from jetnet.dataset import Dataset
from jetnet.image import Image, ImageDatasetConfig
from jetnet.text_detection import TextDetectionModel, TextDetectionSet, TextDetectionModelConfig
from jetnet.utils import make_parent_dir
from jetnet.tensorrt import trt_calib_algo_from_str, Int8CalibAlgo, trt_log_level_from_str, TrtLogLevel
from jetnet.easyocr.easyocr_model import EasyOCRModel
from jetnet.easyocr.easyocr_model_config import EasyOCRModelConfig


class EasyOCRTRTEngineConfig(Config):
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 26
    engine_cache: Optional[str] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 16
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"
    min_shapes: Optional[Sequence[Sequence[int]]] = None
    max_shapes: Optional[Sequence[Sequence[int]]] = None
    opt_shapes: Optional[Sequence[Sequence[int]]] = None
    use_onnx: bool = False
    onnx_opset: int = 11
    log_level: TrtLogLevel = "error"


EASYOCR_TRT_DEFAULT_DETECTOR_CFG = EasyOCRTRTEngineConfig(
    use_onnx=True,
    min_shapes=[(1, 3, 240, 320)],
    opt_shapes=[(1, 3, 720, 1280)],
    max_shapes=[(1, 3, 1080, 1920)],
    int8_calib_algorithm="minmax"
)

EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG = EasyOCRTRTEngineConfig(
    use_onnx=True,
    min_shapes=[(1, 1, 64, 32)],
    opt_shapes=[(1, 1, 64, 320)],
    max_shapes=[(1, 1, 64, 1920)]
)

class EasyOCRTRTModelConfig(TextDetectionModelConfig):
    model: EasyOCRModelConfig
    int8_calib_dataset: Optional[ImageDatasetConfig] = None
    detector_config: Optional[EasyOCRTRTEngineConfig] = EASYOCR_TRT_DEFAULT_DETECTOR_CFG
    recognizer_config: Optional[EasyOCRTRTEngineConfig] = EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG
    
    def _build_trt(self, model, module, cfg: EasyOCRTRTEngineConfig, is_recog=False):
        
        if cfg.engine_cache is not None and os.path.exists(cfg.engine_cache):
            module = TRTModule()
            module.load_state_dict(torch.load(cfg.engine_cache))
            return module
        
        if cfg.int8_mode:
            
            if cfg.int8_calib_cache is None:
                cfg.int8_calib_cache = tempfile.mkdtemp()

            calib_folder = FolderDataset(cfg.int8_calib_cache)
            
            if len(calib_folder) < cfg.int8_num_calib:

                assert self.int8_calib_dataset is not None

                int8_calib_dataset = self.int8_calib_dataset.build()
                i = len(calib_folder)

                pbar = ProgressBar(maxval=cfg.int8_num_calib, widgets=["Generating INT8 calibration data [", Timer(), "] ", Bar(), " (", ETA(), ")"])
                pbar.start()
                with calib_folder.record(module):
                    while len(calib_folder) < cfg.int8_num_calib:
                        x = int8_calib_dataset[i % len(int8_calib_dataset)]
                        model(x)
                        i += 1
                        pbar.update(i)
                pbar.finish()
        else:
            calib_folder = None
        
        module = module.cuda().eval()

        if is_recog:
            class PoolFix(torch.nn.Module):
                def forward(self, x):
                    return torch.mean(x, dim=-1, keepdim=True)
            module.AdaptiveAvgPool = PoolFix()
            data = [torch.randn(cfg.opt_shapes[0]).cuda(), None]
        else:
            data = [torch.randn(cfg.opt_shapes[0]).cuda()]

        module = module.cuda().eval()
        module_trt = torch2trt(
            module,
            data,
            fp16_mode=cfg.fp16_mode,
            int8_mode=cfg.int8_mode,
            max_workspace_size=cfg.max_workspace_size,
            int8_calib_dataset=calib_folder,
            int8_calib_algorithm=trt_calib_algo_from_str(cfg.int8_calib_algorithm),
            min_shapes=cfg.min_shapes,
            max_shapes=cfg.max_shapes,
            opt_shapes=cfg.opt_shapes,
            use_onnx=cfg.use_onnx,
            onnx_opset=cfg.onnx_opset,
            log_level=trt_log_level_from_str(cfg.log_level)
        )

        if cfg.engine_cache is not None:
            make_parent_dir(cfg.engine_cache)
            torch.save(module_trt.state_dict(), cfg.engine_cache)

        return module_trt

    def build(self) -> EasyOCRModel:
        model = self.model.build()

        if self.detector_config is not None:
            det = model._reader.detector.module
            det_trt = self._build_trt(model, det, self.detector_config)

        if self.recognizer_config is not None:
            rec = model._reader.recognizer.module
            rec_trt = self._build_trt(model, rec, self.recognizer_config, is_recog=True)

        if self.detector_config is not None:
            model._reader.detector.module = det_trt
        if self.recognizer_config is not None:
            model._reader.recognizer.module = rec_trt

        return model