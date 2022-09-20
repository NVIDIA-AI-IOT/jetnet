from typing import Literal, Optional, Sequence, Tuple, Literal
import os

import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from typing import Callable

from jetnet.pose import PoseModel, PoseSet, Keypoint, Pose
from jetnet.image import Image

from trt_pose.coco import coco_category_to_topology
from trt_pose.parse_objects import ParseObjects


__all__ = ["TRTPoseModel"]


class TRTPoseModel(PoseModel):


    def __init__(self,
            module: torch.nn.Module,
            keypoints: Sequence[str],
            skeleton: Sequence[Tuple[int, int]],
            input_size: Tuple[int, int],
            device: torch.device
        ):
        self._module = module.to(device).eval()
        self._device = device
        self._skeleton = skeleton
        self._keypoints = keypoints
        coco_skeleton = [[a+1, b+1] for a, b in skeleton]
        self._topology = coco_category_to_topology(
            {"keypoints": self._keypoints, "skeleton": coco_skeleton}
        )
        self._parse_objects = ParseObjects(self._topology)
        self._normalize = Normalize(
            [255.0 * 0.485, 255.0 * 0.456, 255.0 * 0.406],
            [255.0 * 0.229, 255.0 * 0.224, 255.0 * 0.225],
        ).to(self._device)
        self._input_size = input_size

    def get_keypoints(self) -> Sequence[str]:
        return self._keypoints

    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        return self._skeleton

    def __call__(self, x: Image) -> PoseSet:
        with torch.no_grad():
            image = x.pil()
            width, height = image.width, image.height
            data = (
                torch.from_numpy(np.asarray(image))
                .to(self._device)
                .permute(2, 0, 1)
                .float()
            )[None, ...]

            data = F.interpolate(data, size=self._input_size[::-1])

            data = self._normalize(data[0])
            cmap, paf = self._module(data[None, ...])
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

            counts, objects, peaks = self._parse_objects(cmap, paf)
            count = int(counts[0])

            poses = []

            for i in range(count):

                keypoints = []
                obj = objects[0][i]
                c = obj.shape[0]

                for j in range(c):
                    k = int(obj[j])
                    if k >= 0:
                        peak = peaks[0][j][k]
                        kp = Keypoint.construct(
                            x = round(float(peak[1]) * width),
                            y = round(float(peak[0]) * height),
                            index = j,
                            label=self.get_keypoints()[j]
                        )
                        keypoints.append(kp)

                pose = Pose.construct(
                    keypoints=keypoints
                )

                poses.append(pose)

            return PoseSet.construct(poses=poses)