from jetnet.torchvision import RESNET18_IMAGENET_TRT
from jetnet.demo.classification_demo import ClassificationDemoConfig


config = ClassificationDemoConfig(
    model_config=RESNET18_IMAGENET_TRT
)

demo = config.build()

demo.run()