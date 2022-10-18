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

import time
import os
from pydantic import BaseModel

from jetnet.utils import import_object

from jetnet.classification import ClassificationModel
from jetnet.text_detection import TextDetectionModel
from jetnet.detection import DetectionModel
from jetnet.pose import Pose, PoseModel


class ProfileResult(BaseModel):
    fps: float


class Profile(BaseModel):
    num_warmup: int = 10
    num_profile: int = 50
    sleep_interval: float = 0.01

    def run(self, model, dataset) -> ProfileResult:

        dataset_size = len(dataset)
        for i in range(self.num_warmup):
            model(dataset[i % dataset_size])

        elapsed_time = 0.0
        for i in range(self.num_profile):
            image = dataset[i % dataset_size]
            time.sleep(self.sleep_interval)
            t0 = time.perf_counter()
            model(image)
            t1 = time.perf_counter()
            elapsed_time += (t1 - t0)
        fps = self.num_profile / elapsed_time
        result = ProfileResult(fps=fps)
        return result


def register_args(parser):
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--num_warmup', type=str, default=10)
    parser.add_argument('--num_profile', type=int, default=50)
    parser.add_argument('--sleep_interval', type=float, default=0.01)


def run_args(args):
    from jetnet.profile.profile import Profile
    from jetnet.profile.classification import ClassificationProfile
    from jetnet.profile.detection import DetectionProfile
    from jetnet.profile.pose import PoseProfile
    from jetnet.profile.text_detection import TextDetectionProfile


    
    model = import_object(args.model)
    dataset = import_object(args.dataset)
    model = model.build()
    dataset = dataset.build()

    if issubclass(model.__class__, ClassificationModel):
        app_cls = ClassificationProfile
    elif issubclass(model.__class__, DetectionModel):
        app_cls = DetectionProfile
    elif issubclass(model.__class__, PoseModel):
        app_cls = PoseProfile
    elif issubclass(model.__class__, TextDetectionModel):
        app_cls = TextDetectionProfile
    else:
        app_cls = Profile
    
    profile = app_cls(
        num_warmup=args.num_warmup,
        num_profile=args.num_profile,
        sleep_interval=args.sleep_interval
    )

    result = profile.run(model, dataset)

    print(result.json(indent=2))

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)