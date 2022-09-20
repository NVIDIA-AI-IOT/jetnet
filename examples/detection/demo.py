from jetnet.yolox import YOLOX_TINY_TRT_FP16
from jetnet.demo.detection_demo import DetectionDemoConfig


config = DetectionDemoConfig(
    model_config=YOLOX_TINY_TRT_FP16
)

demo = config.build()

demo.run()