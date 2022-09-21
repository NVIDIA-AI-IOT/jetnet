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

from jetnet.image import (
    RemoteImageFolder,
)

__all__ = [
    "COCO2017_VAL_IMAGES",
    "COCO2017_TRAIN_IMAGES",
    "COCO2017_TEST_IMAGES"
]


COCO2017_VAL_IMAGES = RemoteImageFolder(
    path="data/coco/val2017",
    zip_url="http://images.cocodataset.org/zips/val2017.zip",
    zip_file="data/coco/val2017.zip",
    zip_folder="val2017"
)

COCO2017_TRAIN_IMAGES = RemoteImageFolder(
    path="data/coco/train2017",
    zip_url="http://images.cocodataset.org/zips/train2017.zip",
    zip_file="data/datasets/coco/train2017.zip",
    zip_folder="train2017"
)

COCO2017_TEST_IMAGES = RemoteImageFolder(
    path="data/coco/test2017",
    zip_url="http://images.cocodataset.org/zips/test2017.zip",
    zip_file="data/datasets/coco/test2017.zip",
    zip_folder="test2017"
)

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
