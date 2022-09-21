import os
import time
from typing import Tuple
from .profile import ProfileResult, Profile, register_args, run_args


class DetectionProfileResult(ProfileResult):
    avg_image_area: float
    avg_num_detections: float


def profile_detection(
        model, 
        dataset, 
        num_profile=50, 
        num_warmup=10, 
        sleep_interval=0.01
    ) -> DetectionProfileResult:
    dataset_size = len(dataset)
    for i in range(num_warmup):
        model(dataset[i % dataset_size])
    image_areas = []
    num_detections = []
    elapsed_time = 0.0
    for i in range(num_profile):
        image = dataset[i % dataset_size]
        image_areas.append(image.width * image.height)
        time.sleep(sleep_interval)
        t0 = time.monotonic()
        output = model(image)
        t1 = time.monotonic()
        elapsed_time += (t1 - t0)
        num_detections.append(len(output.detections))

    fps = num_profile / elapsed_time
    result = DetectionProfileResult(
        fps=fps, 
        avg_image_area=sum(image_areas) / len(image_areas),
        avg_num_detections=sum(num_detections) / len(num_detections)
    )
    return result


class DetectionProfile(Profile):

    def build(self) -> DetectionProfileResult:
        return profile_detection(
            self.model.build(),
            self.dataset.build(),
            self.num_profile,
            self.num_warmup,
            self.sleep_interval
        )
