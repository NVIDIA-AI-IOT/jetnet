from jetnet.classification import ClassificationModel, ClassificationModelConfig
from jetnet.config import Config
from jetnet.imagenet import IMAGENET_LABELS

from typing import Tuple, Literal, Sequence


class TorchvisionModelConfig(ClassificationModelConfig):
    
    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenet_v2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201"
    ]

    input_size: Tuple[int, int] = (224, 224)
    pretrained: bool = False
    labels: Sequence[str] = IMAGENET_LABELS
    device: Literal["cpu", "cuda"] = "cuda"

    def build(self) -> ClassificationModel:

        import torch
        import torchvision
        from jetnet.torchvision.torchvision_model import TorchvisionModel

        device = torch.device(self.device)
        module = getattr(torchvision.models, self.name)(pretrained=self.pretrained).to(device).eval()
        return TorchvisionModel(
            module,
            device,
            self.labels,
            self.input_size
        )
