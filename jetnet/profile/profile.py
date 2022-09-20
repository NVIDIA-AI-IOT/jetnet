import time
import os
from pydantic import BaseModel
from jetnet.config import Config

from jetnet.utils import import_object

from jetnet.classification import ClassificationModelConfig
from jetnet.text_detection import TextDetectionModelConfig
from jetnet.detection import DetectionModelConfig
from jetnet.pose import Pose, PoseModelConfig

class ProfileResult(BaseModel):
    fps: float


class ProfileConfig(Config):
    model_config: Config # must be model
    dataset_config: Config # must be dataset
    num_warmup: int = 10
    num_profile: int = 50
    sleep_interval: float = 0.01

    def build(self) -> ProfileResult:
        model = self.model_config.build()
        dataset = self.dataset_config.build()

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])

        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            time.sleep(self.sleep_interval)
            t0 = time.monotonic()
            model(image)
            t1 = time.monotonic()
            elapsed_time += (t1 - t0)
        fps = self.num_profile / elapsed_time
        result = ProfileResult(fps=fps)
        return result


def register_args(parser):
    parser.add_argument('model_config', type=str)
    parser.add_argument('dataset_config', type=str)
    parser.add_argument('--num_warmup', type=str, default=10)
    parser.add_argument('--num_profile', type=int, default=50)
    parser.add_argument('--sleep_interval', type=int, default=0.01)


def run_args(args):
    from jetnet.profile.profile import ProfileConfig
    from jetnet.profile.classification import ClassificationProfileConfig
    from jetnet.profile.detection import DetectionProfileConfig
    from jetnet.profile.pose import PoseProfileConfig
    from jetnet.profile.text_detection import TextDetectionProfileConfig


    
    model_config = import_object(args.model_config)
    dataset_config = import_object(args.dataset_config)

    if issubclass(model_config.__class__, ClassificationModelConfig):
        app_cls = ClassificationProfileConfig
    elif issubclass(model_config.__class__, DetectionModelConfig):
        app_cls = DetectionProfileConfig
    elif issubclass(model_config.__class__, PoseModelConfig):
        app_cls = PoseProfileConfig
    elif issubclass(model_config.__class__, TextDetectionModelConfig):
        app_cls = TextDetectionProfileConfig
    else:
        app_cls = ProfileConfig
    print(app_cls)
    config = app_cls(
        model_config=model_config,
        dataset_config=dataset_config,
        num_warmup=args.num_warmup,
        num_profile=args.num_profile,
        sleep_interval=args.sleep_interval
    )

    result = config.build()

    print(result.json(indent=2))

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)