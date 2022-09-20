import os
import time
from typing import Tuple
from .profile import ProfileResult, ProfileConfig, register_args, run_args


class TextDetectionProfileResult(ProfileResult):
    avg_image_area: float
    avg_num_detections: float
    avg_num_characters: float


class TextDetectionProfileConfig(ProfileConfig):

    def build(self) -> TextDetectionProfileResult:
        model = self.model_config.build()
        dataset = self.dataset_config.build()

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])
        image_areas = []
        num_detections = []
        num_characters = []
        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            image_areas.append(image.pil().width * image.pil().height)
            time.sleep(self.sleep_interval)
            t0 = time.monotonic()
            output = model(image)
            t1 = time.monotonic()
            elapsed_time += (t1 - t0)
            num_detections.append(len(output.detections))
            num_characters.append(sum(len(p.text) for p in output.detections))

        fps = self.num_profile / elapsed_time
        result = TextDetectionProfileResult(
            fps=fps, 
            avg_image_area=sum(image_areas) / len(image_areas),
            avg_num_detections=sum(num_detections) / len(num_detections),
            avg_num_characters=sum(num_characters) / len(num_characters)
        )
        return result

