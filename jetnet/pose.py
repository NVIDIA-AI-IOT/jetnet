from pydantic import BaseModel
from jetnet.keypoint import Keypoint
from jetnet.model import Model
from jetnet.image import Image


from abc import abstractmethod
from typing import Sequence, Tuple


class Pose(BaseModel):
    keypoints: Sequence[Keypoint]


class PoseSet(BaseModel):
    poses: Sequence[Pose]


class PoseModel(Model[Image, PoseSet]):
    
    @abstractmethod
    def get_keypoints(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        raise NotImplementedError

