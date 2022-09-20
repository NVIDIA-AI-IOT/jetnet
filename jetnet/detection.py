from abc import abstractmethod

from jetnet.polygon import Polygon
from jetnet.classification import Classification
from jetnet.model import Model
from jetnet.image import Image
from jetnet.config import Config
from pydantic import BaseModel


from typing import Optional, Sequence


class Detection(BaseModel):

    boundary: Polygon
    classification: Optional[Classification] = None


class DetectionSet(BaseModel):
    detections: Sequence[Detection]


class DetectionModel(Model[Image, DetectionSet]):
    
    @abstractmethod
    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError


class DetectionModelConfig(Config[DetectionModel]):
    pass