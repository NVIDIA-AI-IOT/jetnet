# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Tuple
from .profile import ProfileResult, Profile, register_args, run_args


class PoseProfileResult(ProfileResult):
    avg_image_area: float
    avg_num_poses: float
    avg_num_keypoints: float


class PoseProfile(Profile):

    def build(self) -> PoseProfileResult:
        model = self.model.build()
        dataset = self.dataset.build()

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])
        image_areas = []
        num_poses = []
        num_keypoints = []
        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            image_areas.append(image.width * image.height)
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