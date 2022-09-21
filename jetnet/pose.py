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


from pydantic import BaseModel
from jetnet.keypoint import Keypoint
from jetnet.model import Model
from jetnet.image import Image


from abc import abstractmethod
from typing import Sequence, Tuple


class Pose(BaseModel):
    keypoints: Sequence[Keypoint]


class PoseSet(BaseModel):
    poses: Sequence[Pose]


class PoseModel(Model[Image, PoseSet]):
    
    @abstractmethod
    def get_keypoints(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        raise NotImplementedError

