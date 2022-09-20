from jetnet.profile.profile_pose import PoseProfileConfig
from jetnet.trt_pose import RESNET18_BODY_224X224_TRT_FP16
from jetnet.coco import COCO2017_VAL_IMAGES


config = PoseProfileConfig(
    model_config=RESNET18_BODY_224X224_TRT_FP16,
    dataset_config=COCO2017_VAL_IMAGES
)

result = config.build()

print(result.json(indent=2))