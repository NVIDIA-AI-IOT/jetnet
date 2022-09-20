from jetnet.easyocr import EASYOCR_EN_TRT_FP16
from jetnet.demo.text_detection_demo import TextDetectionDemoConfig


config = TextDetectionDemoConfig(
    model_config=EASYOCR_EN_TRT_FP16
)

demo = config.build()

demo.run()