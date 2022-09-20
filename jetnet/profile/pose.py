import os
import time
from typing import Tuple
from .profile import ProfileResult, ProfileConfig, register_args, run_args


class PoseProfileResult(ProfileResult):
    avg_image_area: float
    avg_num_poses: float
    avg_num_keypoints: float


class PoseProfileConfig(ProfileConfig):

    def build(self) -> PoseProfileResult:
        model = self.model_config.build()
        dataset = self.dataset_config.build()

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])
        image_areas = []
        num_poses = []
        num_keypoints = []
        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            image_areas.append(image.pil().width * image.pil().height)
            time.sleep(self.sleep_interval)
            t0 = time.monotonic()
            output = model(image)
            t1 = time.monotonic()
            elapsed_time += (t1 - t0)
            num_poses.append(len(output.poses))
            num_keypoints.append(sum(len(p.keypoints) for p in output.poses))

        fps = self.num_profile / elapsed_time
        result = PoseProfileResult(
            fps=fps, 
            avg_image_area=sum(image_areas) / len(image_areas),
            avg_num_poses=sum(num_poses) / len(num_poses),
            avg_num_keypoints=sum(num_keypoints) / len(num_keypoints)
        )
        return result



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(PoseProfileConfig, args)