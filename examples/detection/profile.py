from jetnet.profile.detection_profile import DetectionProfileConfig
from jetnet.yolox import YOLOX_TINY_TRT_FP16
from jetnet.coco import COCO2017_VAL_IMAGES

config = DetectionProfileConfig(
    model_config=YOLOX_TINY_TRT_FP16,
    dataset_config=COCO2017_VAL_IMAGES
)

result = config.build()

print(result.json(indent=2))