from jetnet.config import Config
from jetnet.pose import PoseModel, PoseModelConfig

from typing import Literal, Sequence, Tuple, Optional


class TRTPoseModelConfig(PoseModelConfig):

    model: Literal[
        "resnet18_baseline_att",
        "densenet121_baseline_att"
    ]
    keypoints: Sequence[str]
    skeleton: Sequence[Tuple[int, int]]
    input_size: Tuple[int, int]
    weights_url: Optional[str] = None
    weights_path: Optional[str] = None
    device: Literal["cpu", "cuda"] = "cuda"

    def build(self) -> PoseModel:
        
        import torch
        import os
        from jetnet.trt_pose.trt_pose_model import TRTPoseModel
        from jetnet.utils import make_parent_dir, download
        from trt_pose.models import MODELS

        module = MODELS[self.model](len(self.keypoints), 2 * len(self.skeleton))

        if self.weights_path is not None:
            if not os.path.exists(self.weights_path):
                make_parent_dir(self.weights_path)
                download(self.weights_url, self.weights_path)
            module.load_state_dict(torch.load(self.weights_path))

        device = torch.device(self.device)
        module = module.to(device).eval()
        
        return TRTPoseModel(module, self.keypoints, self.skeleton, self.input_size, device)