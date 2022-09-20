import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms
import torchvision

from torch2trt import torch2trt, trt, TRTModule
from torch2trt.dataset import FolderDataset

from progressbar import ProgressBar, Bar, Timer, ETA
from pydantic import PrivateAttr
from typing import Tuple, Sequence, Literal, Optional

from jetnet.imagenet import IMAGENET_LABELS
from jetnet.image import Image, ImageDataset
from jetnet.classification import Classification, ClassificationModel
from jetnet.imagenet import IMAGENET_LABELS
from jetnet.dataset import Dataset
from jetnet.tensorrt import trt_calib_algo_from_str, Int8CalibAlgo
from jetnet.utils import make_parent_dir


class TorchvisionModel(ClassificationModel):

    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenet_v2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201"
    ]

    input_size: Tuple[int, int] = (224, 224)
    pretrained: bool = False
    labels: Sequence[str] = IMAGENET_LABELS
    device: Literal["cpu", "cuda"] = "cuda"

    _module = PrivateAttr()
    _device = PrivateAttr()
    _normalize = PrivateAttr()
    
    def init(self):
        self._device = torch.device(self.device)
        module = getattr(torchvision.models, self.name)(pretrained=self.pretrained)
        self._module = module.to(self._device).eval()
        self._normalize = torchvision.transforms.Normalize(
            [255.0 * 0.485, 255.0 * 0.456, 255.0 * 0.406],
            [255.0 * 0.229, 255.0 * 0.224, 255.0 * 0.225],
        ).to(self._device)

    def get_labels(self):
        return self.labels

    def __call__(self, x: Image) -> Classification:
        with torch.no_grad():
            tensor = (
                torch.from_numpy(np.array(x))
                .to(self._device)
                .permute(2, 0, 1)
                .float()
            )
            tensor = F.interpolate(tensor[None, ...], size=self.input_size[::-1])
            tensor = self._normalize(tensor[0])
            output = self._module(tensor[None, ...]).cpu()
            index = int(torch.argmax(output[0]))
            score = float(output[0, index])
            label = self.get_labels()[index]
            return Classification.construct(
                index=index,
                score=score,
                label=label
            )


class TorchvisionModelTRT(ClassificationModel):

    model: TorchvisionModel
    int8_mode: bool = False
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 25
    engine_cache: Optional[str] = None
    int8_calib_dataset: Optional[ImageDataset] = None
    int8_calib_cache: Optional[str] = None
    int8_num_calib: int = 1
    int8_calib_algorithm: Int8CalibAlgo = "entropy_2"

    def init(self):

        model = self.model.build()

        if self.engine_cache is not None and os.path.exists(self.engine_cache):
            module = TRTModule()
            module.load_state_dict(torch.load(self.engine_cache))
            model._module = module
            self.model = model
            return self
        
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

        data = torch.randn((1, 3) + model.input_size[::-1]).to(model._device)

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
        self.model = model
        return self

    def get_labels(self):
        return self.model.get_labels()

    def __call__(self, x: Image) -> Classification:
        return self.model(x)