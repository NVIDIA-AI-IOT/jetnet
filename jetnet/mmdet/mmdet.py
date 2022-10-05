from multiprocessing.sharedctypes import Value
import yaml
import os
from mmdet.apis import init_detector, inference_detector
from jetnet.utils import download, make_parent_dir
from jetnet.config import Config
from torch2trt import torch2trt
from typing import Optional
import torch


def find_mm_models(root):
    models = []
    root_model_index_path = os.path.join(root, 'model-index.yml')
    with open(root_model_index_path, 'r') as f:
        root_model_index = yaml.load(f)
    for model_index_path in root_model_index['Import']:
        
        model_index_path = os.path.join(root, model_index_path)
        
        with open(model_index_path, 'r') as f:
            model_index = yaml.load(f)
        for model in model_index['Models']:
            models.append(model)
            
    return models


def find_mmdet_models():
    return find_mm_models(os.environ['MMDET_DIR'])


def find_mmocr_models():
    return find_mm_models(os.environ['MMOCR_DIR'])

def is_task(model, task):
    for res in model['Results']:
        if res['Task'] == task:
            return True
    return False
    return len(list(result for result in model['Results'] if result))


INSTANCE_SEGMENTATION = 'Instance Segmentation'
OBJECT_DETECTION = 'Object Detection'


def find_mmdet_instance_segmentation_models():
    return [m for m in find_mmdet_models() if is_task(m, INSTANCE_SEGMENTATION)]


def find_mmdet_object_detection_models():
    return [m for m in find_mmdet_models() if is_task(m, OBJECT_DETECTION)]
    
def init_detector_by_name(name, mmdet_dir=None, weights_dir="data/mmdet"):
    
    if mmdet_dir is None:
        mmdet_dir = os.environ.get('MMDET_DIR')
    assert mmdet_dir is not None
    
    model = next(model for model in find_mmdet_models() if model['Name'] == name)
    config = os.path.join(os.environ['MMDET_DIR'], model['Config'])
    weights_url = model['Weights']
    weights_path = os.path.join(weights_dir, os.path.basename(weights_url))
    if not os.path.exists(weights_path):
        make_parent_dir(weights_path)
        download(weights_url, weights_path)
    
    return init_detector(config, weights_path)

def get_feat_strides(cfg):
    return cfg['model']['roi_head']['bbox_roi_extractor']['featmap_strides']

def get_neck_input_shapes(cfg, shape):
    height, width = shape
    strides = get_feat_strides(cfg)
    in_channels = cfg['model']['neck']['in_channels']
    return [[1, ic, height // s, width // s] for ic, s in zip(in_channels, strides)]

def get_bbox_max_input_shapes(cfg):
    roi_size = cfg['model']['roi_head']['bbox_head']['roi_feat_size']
    in_channels = cfg['model']['roi_head']['bbox_head']['in_channels']
    max_count = cfg['model']['test_cfg']['rpn']['max_per_img']
    return [[max_count, in_channels, roi_size, roi_size]]

def get_bbox_min_input_shapes(cfg):
    shape = get_bbox_max_input_shapes(cfg)
    shape[0][0] = 1
    return shape

def get_mask_max_input_shapes(cfg):
    max_count = cfg['model']['test_cfg']['rcnn']['max_per_img']
    roi_size = cfg['model']['roi_head']['mask_roi_extractor']['roi_layer']['output_size']
    in_channels = cfg['model']['roi_head']['mask_head']['in_channels']
    return [[max_count, in_channels, roi_size, roi_size]]

def get_mask_min_input_shapes(cfg):
    shape = get_mask_max_input_shapes(cfg)
    shape[0][0] = 1
    return shape

def get_shapes(cfg, min_shape, max_shape, opt_shape):
    min_shape = list(min_shape)
    max_shape = list(max_shape)
    opt_shape = list(opt_shape)
    shapes = {
        'bbox': {
            'min': get_bbox_min_input_shapes(cfg),
            'max': get_bbox_max_input_shapes(cfg),
            'opt': get_bbox_min_input_shapes(cfg)
        },
        'neck': {
            'min': get_neck_input_shapes(cfg, min_shape),
            'max': get_neck_input_shapes(cfg, max_shape),
            'opt': get_neck_input_shapes(cfg, opt_shape)
        },
        'backbone': {
            'min': [[1, 3] + min_shape],
            'max': [[1, 3] + max_shape],
            'opt': [[1, 3] + opt_shape]
        },
        'mask': {
            'min': get_mask_min_input_shapes(cfg),
            'max': get_mask_max_input_shapes(cfg),
            'opt': get_mask_min_input_shapes(cfg)
        }
    }
    return shapes

def make_inputs(desc):
    return [torch.randn(d).cuda() for d in desc['opt']]

def mmdet_mask_rcnn_build_torch2trt_modules(det, min_shape, max_shape, opt_shape, fp16_mode=False):
    backbone = det.backbone
    neck = det.neck
    bbox = det.roi_head.bbox_head
    mask = det.roi_head.mask_head
    
    shapes = get_shapes(det.cfg, min_shape, max_shape, opt_shape)
    
    def _run_torch2trt(module, desc, fp16_mode, expand=True):
        print(f"Optimizing... {desc}")
        inputs = make_inputs(desc)
        if not expand:
            inputs = [inputs]
            min_shapes = [desc['min']]
            max_shapes = [desc['max']]
            opt_shapes = [desc['opt']]
        else:
            min_shapes = desc['min']
            max_shapes = desc['max']
            opt_shapes = desc['opt']
            
        return torch2trt(
            module,
            inputs,
            fp16_mode=fp16_mode,
            use_onnx=True,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            opt_shapes=opt_shapes,
            onnx_opset=11
        )
    
    backbone_trt = _run_torch2trt(backbone, shapes['backbone'], fp16_mode)
    neck_trt = _run_torch2trt(neck, shapes['neck'], fp16_mode, expand=False)
    bbox_trt = _run_torch2trt(bbox, shapes['bbox'], fp16_mode)
    mask_trt = _run_torch2trt(mask, shapes['mask'], fp16_mode)
    return {
        "backbone": backbone_trt,
        "neck": neck_trt,
        "bbox": bbox_trt,
        "mask": mask_trt
    }


def mmdet_mask_rcnn_inject_torch2trt_modules(det, modules):
    det.backbone = modules['backbone']
    det.neck = modules['neck']
    det.bbox = modules['bbox']
    det.mask = modules['mask']
    return det

def _trt_forward(module, x):
    return module._trt(x)

class _MMDet:

    def __init__(self, detector):
        self.detector = detector

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


class MMDet(Config[_MMDet]):

    config: str
    weights: str
    weights_url: Optional[str] = None

    def build(self):
        
        config_path = os.path.expandvars(self.config)
        weights_path = os.path.expandvars(self.weights)

        if not os.path.exists(weights_path):
            assert self.weights_url is not None
            make_parent_dir(weights_path)
            download(self.weights_url, weights_path)

        return _MMDet(init_detector(config_path, weights_path))


