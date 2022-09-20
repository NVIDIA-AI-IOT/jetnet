from abc import abstractmethod
from pydantic import BaseModel
from jetnet.image import Image
from jetnet.model import Model
from jetnet.config import Config


from typing import Optional, Sequence


class Classification(BaseModel):
    index: int
    label: Optional[str] = None
    score: Optional[float] = None


class ClassificationModel(Model[Image, Classification]):
    
    @abstractmethod
    def get_labels(self) -> Sequence[str]:
        raise NotImplementedError


class ClassificationModelConfig(Config[ClassificationModel]):
    pass