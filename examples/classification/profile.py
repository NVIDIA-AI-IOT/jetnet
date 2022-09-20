from jetnet.profile.profile_classification import ClassificationProfileConfig
from jetnet.torchvision import RESNET18_IMAGENET_TRT_FP16
from jetnet.coco import COCO2017_VAL_IMAGES


config = ClassificationProfileConfig(
    model_config=RESNET18_IMAGENET_TRT_FP16,
    dataset_config=COCO2017_VAL_IMAGES
)

result = config.build()

print(result.json(indent=2))