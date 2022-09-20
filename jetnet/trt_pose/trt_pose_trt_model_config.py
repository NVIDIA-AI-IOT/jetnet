from jetnet.config import Config
from jetnet.image import Image, ImageDatasetConfig
from jetnet.dataset import Dataset
from jetnet.trt_pose.trt_pose_model_config import TRTPoseModelConfig
from jetnet.tensorrt import Int8CalibAlgo
from jetnet.pose import PoseModel, PoseModelConfig

from typing import Optional


__all__ = ["TRTPoseTRTModelConfig"]


class TRTPoseTRTModelConfig(PoseModelConfig):

    model: TRTPoseModelConfig
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 25
    engine_cache: Optional[str] = None
    int8_calib_dataset: Optional[ImageDatasetConfig] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 1
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"

        
    def build(self) -> PoseModel:

        import tempfile
        import os
        import torch
        from torch2trt import torch2trt
        from torch2trt.dataset import FolderDataset
        from jetnet.utils import make_parent_dir
        from jetnet.tensorrt import trt_calib_algo_from_str
        from progressbar import Timer, Bar, ETA, ProgressBar
        from torch2trt import TRTModule

        model = self.model.build()

        if self.engine_cache is not None and os.path.exists(self.engine_cache):
            module = TRTModule()
            module.load_state_dict(torch.load(self.engine_cache))
            model._module = module
            return model
        
        if self.int8_mode:
            
            if self.int8_calib_cache is None:
                self.int8_calib_cache = tempfile.mkdtemp()

            calib_folder = FolderDataset(self.int8_calib_cache)
            
            if len(calib_folder) < self.int8_num_calib:

                assert self.int8_calib_dataset is not None

                int8_calib_dataset = self.int8_calib_dataset.build()
                i = len(calib_folder)

                pbar = ProgressBar(maxval=self.int8_num_calib, widgets=["Generating INT8 calibration data [", Timer(), "] ", Bar(), " (", ETA(), ")"])
                pbar.start()
                with calib_folder.record(model._module):
                    while len(calib_folder) < self.int8_num_calib:
                        x = int8_calib_dataset[i % len(int8_calib_dataset)]
                        model(x)
                        i += 1
                        pbar.update(i)
                pbar.finish()
        else:
            calib_folder = None

        data = torch.randn((1, 3) + self.model.input_size[::-1]).to(model._device)

        module = model._module.to(model._device).eval()

        module_trt = torch2trt(
            module,
            [data],
            fp16_mode=self.fp16_mode,
            int8_mode=self.int8_mode,
            max_workspace_size=self.max_workspace_size,
            int8_calib_dataset=calib_folder,
            int8_calib_algorithm=trt_calib_algo_from_str(self.int8_calib_algorithm)
        )

        if self.engine_cache is not None:
            make_parent_dir(self.engine_cache)
            torch.save(module_trt.state_dict(), self.engine_cache)

        model._module = module_trt

        return model
