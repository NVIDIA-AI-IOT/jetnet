import os
import time
from typing import Tuple
from .profile import ProfileResult, Profile, register_args, run_args


class ClassificationProfileResult(ProfileResult):
    avg_image_area: float


class ClassificationProfile(Profile):

    def build(self) -> ClassificationProfileResult:
        model = self.model.build()
        dataset = self.dataset.build()

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])
        image_areas = []
        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            image_areas.append(image.width * image.height)
            time.sleep(self.sleep_interval)
            t0 = time.monotonic()
            model(image)
            t1 = time.monotonic()
            elapsed_time += (t1 - t0)
        fps = self.num_profile / elapsed_time
        avg_image_size = sum(image_areas) / len(image_areas)
        result = ClassificationProfileResult(fps=fps, avg_image_area=avg_image_size)
        return result
