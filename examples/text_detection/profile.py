from jetnet.profile.profile_text_detection import TextDetectionProfileConfig
from jetnet.easyocr import EASYOCR_EN_TRT_FP16
from jetnet.textocr import TEXTOCR_TEST_IMAGES


config = TextDetectionProfileConfig(
    model_config=EASYOCR_EN_TRT_FP16,
    dataset_config=TEXTOCR_TEST_IMAGES
)

result = config.build()

print(result.json(indent=2))
