from jetnet.easyocr.easyocr_model import EasyOCRModel
from jetnet.text_detection import TextDetectionModelConfig
from typing import Sequence
from easyocr import Reader


class EasyOCRModelConfig(TextDetectionModelConfig):

    lang_list: Sequence[str]

    def build(self) -> EasyOCRModel:
        reader = Reader(lang_list=self.lang_list)
        return EasyOCRModel(reader)