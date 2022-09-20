from typing import Literal, Tuple, Sequence, Optional
from jetnet.detection import DetectionModel, DetectionModelConfig
from jetnet.config import Config
from jetnet.coco import COCO_CLASSES


class YOLOXModelConfig(DetectionModelConfig):

    exp: Literal["yolox_l", "yolox_m", "yolox_nano", "yolox_s", "yolox_tiny", "yolox_x"]
    input_size: Tuple[int, int]
    labels: Sequence[str]
    conf_thresh: Optional[float] = 0.3
    nms_thresh: Optional[float] = 0.3
    device: Literal["cpu", "cuda"] = "cuda"
    weights_path: Optional[str] = None
    weights_url: Optional[str] = None
    

    def build(self) -> DetectionModel:

        import torch
        import os
        from jetnet.utils import download, make_parent_dir
        from jetnet.yolox.yolox_model import YOLOXModel

        device = torch.device(self.device)

        exp = self.get_exp()
        exp.test_conf = self.conf_thresh
        exp.nmsthre = self.nms_thresh
        exp.test_size = self.input_size[::-1] # YOLOX uses (h, w), we use (w, h)

        module = exp.get_model()

        if self.weights_path is not None:
            if not os.path.exists(self.weights_path):
                make_parent_dir(self.weights_path)
                if self.weights_url is None:
                    self.weights_url = self._get_weights_url()
                download(self.weights_url, self.weights_path)
            module.load_state_dict(torch.load(self.weights_path)["model"])

        module = module.to(device).eval()

        model = YOLOXModel(
            module,
            device,
            self.input_size,
            self.labels,
            self.conf_thresh,
            self.nms_thresh
        )

        return model

    def _get_weights_url(self):
        if self.exp == "yolox_l":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
        elif self.exp == "yolox_m":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"
        elif self.exp == "yolox_s":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
        elif self.exp == "yolox_tiny":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
        elif self.exp == "yolox_nano":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
        elif self.exp == "yolox_x":
            return "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"
        else:
            raise KeyError("Weights not found.")

    def get_exp(self):
        if self.exp == "yolox_l":
            from .exps.yolox_l import Exp

            return Exp()
        elif self.exp == "yolox_m":
            from .exps.yolox_m import Exp

            return Exp()
        elif self.exp == "yolox_s":
            from .exps.yolox_s import Exp

            return Exp()
        elif self.exp == "yolox_tiny":
            from .exps.yolox_tiny import Exp

            return Exp()
        elif self.exp == "yolox_nano":
            from .exps.yolox_nano import Exp

            return Exp()
        elif self.exp == "yolox_x":
            from .exps.yolox_x import Exp

            return Exp()
        else:
            raise KeyError("Experiment not found.")

