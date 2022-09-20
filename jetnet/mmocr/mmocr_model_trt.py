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
from jetnet.mmocr.mmocr_model import MMOCRModelConfig, MMOCRModel
from jetnet.tensorrt import Torch2trtConfig

__all__ = [
    "MMOCRTRT"
]


class MMOCRTRTModelConfig(TextDetectionModelConfig):
    model: MMOCRModelConfig
    int8_calib_dataset: Optional[ImageDatasetConfig] = None
    detector_config: Optional[Torch2trtConfig] = None
    recognizer_config: Optional[Torch2trtConfig] = None
    
    def _build_trt(self, model, module, cfg: Torch2trtConfig):

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
        
        module.cuda().eval()

        data = [torch.randn(cfg.opt_shapes[0]).cuda()]

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

    def build(self) -> MMOCRModel:
        model = self.model.build()

        if self.detector_config is not None:
            det = model._mmocr.detect_model.backbone
            det_trt = self._build_trt(model, det, self.detector_config)

        if self.recognizer_config is not None:
            rec = model._mmocr.recog_model.backbone
            rec_trt = self._build_trt(model, rec, self.recognizer_config)

        if self.detector_config is not None:
            model._mmocr.detect_model.backbone = det_trt
        if self.recognizer_config is not None:
            model._mmocr.recog_model.backbone = rec_trt

        return model