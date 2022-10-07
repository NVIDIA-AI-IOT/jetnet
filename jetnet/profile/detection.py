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

    def run(self, model, dataset) -> DetectionProfileResult:
        return profile_detection(
            model,
            dataset,
            self.num_profile,
            self.num_warmup,
            self.sleep_interval
        )
