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


from jetnet.profile.classification import ClassificationProfile
from jetnet.torchvision import RESNET18_IMAGENET_TRT_FP16
from jetnet.coco import COCO2017_VAL_IMAGES


config = ClassificationProfile(
    model=RESNET18_IMAGENET_TRT_FP16,
    dataset=COCO2017_VAL_IMAGES
)

result = config.build()

print(result.json(indent=2))