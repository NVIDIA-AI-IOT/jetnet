from pydantic import BaseModel
from jetnet.polygon import Polygon
from jetnet.model import Model
from jetnet.image import Image


from typing import Optional, Sequence


class TextDetection(BaseModel):
    boundary: Polygon
    text: Optional[str]
    score: Optional[float]


class TextDetectionSet(BaseModel):
    detections: Sequence[TextDetection]


class TextDetectionModel(Model[Image, TextDetectionSet]):
    pass

