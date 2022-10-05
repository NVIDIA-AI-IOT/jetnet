from multiprocessing.sharedctypes import Value
import yaml
import numpy as np
import os
from mmdet.apis import init_detector, inference_detector
from jetnet.utils import download, make_parent_dir
from jetnet.config import Config
from jetnet.detection import Detection, DetectionSet
from jetnet.binary_mask import BinaryMask
from jetnet.polygon import Polygon
from jetnet.point import Point
from jetnet.classification import Classification
from torch2trt import torch2trt
from typing import Optional, Sequence
import torch
import cv2
from typing import Any
import mmcv



def _trt_forward(module, x):
    return module._trt(x)

class _MMDet:

    def __init__(self, detector, labels):
        self.detector = detector
        self.labels = labels

    def get_labels(self):
        return self.labels

    def module_names(self):
        return ["backbone", "neck", "bbox", "mask"]
    
    def get_module(self, name) -> torch.nn.Module:
        if name == 'backbone':
            return self.detector.backbone
        elif name == 'neck':
            return self.detector.neck
        elif name == 'bbox':
            return self.detector.roi_head.bbox_head
        elif name == 'mask':
            return self.detector.roi_head.mask_head
        else:
            raise ValueError("Invalid module name")

    def set_module(self, name, value):
        if name == 'backbone':
            self.detector.backbone = value
        elif name == 'neck':
            self.detector.neck = value
        elif name == 'bbox':
            self.detector.roi_head.bbox_head._trt = value
            self.detector.roi_head.bbox_head.forward = _trt_forward.__get__(self.detector.roi_head.bbox_head) # hack to allow original module methods
        elif name == 'mask':
            self.detector.roi_head.mask_head._trt = value
            self.detector.roi_head.mask_head.forward = _trt_forward.__get__(self.detector.roi_head.mask_head)
        else:
            raise ValueError("Invalid module name")

    def __call__(self, x):

        image = x
        data = np.array(image)

        # RGB -> BGR
        if image.mode == "RGB":
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        output_raw = inference_detector(self.detector, data)

        bboxes, masks = output_raw
        
        detections = []
        for label_idx in range(len(bboxes)):
            label = self.get_labels()[label_idx]
            for i in range(len(bboxes[label_idx])):
                bbox = bboxes[label_idx][i]
                x0, y0, x1, y1, score = bbox
                x0, y0, x1, y1, score = int(x0), int(y0), int(x1), int(y1), float(score)

                detection = Detection.construct(
                    boundary=Polygon.construct(points=[
                        Point.construct(x=x0, y=y0),
                        Point.construct(x=x0, y=y1),
                        Point.construct(x=x1, y=y1),
                        Point.construct(x=x1, y=y0)
                    ]),
                    classification=Classification.construct(
                        index=label_idx,
                        label=label,
                        score=score
                    ),
                    mask=BinaryMask.from_numpy(masks[label_idx][i])
                )
                detections.append(detection)

        return DetectionSet.construct(detections=detections)

class MMDet(Config[_MMDet]):

    config: str
    weights: str
    weights_url: Optional[str] = None
    labels: Sequence[str]
    config_options: Optional[Any] = None

    def build(self):
        
        config_path = os.path.expandvars(self.config)
        config = mmcv.Config.fromfile(config_path)
        config.merge_from_dict(self.config_options)

        weights_path = os.path.expandvars(self.weights)

        if not os.path.exists(weights_path):
            assert self.weights_url is not None
            make_parent_dir(weights_path)
            download(self.weights_url, weights_path)

        return _MMDet(init_detector(config, weights_path), self.labels)


