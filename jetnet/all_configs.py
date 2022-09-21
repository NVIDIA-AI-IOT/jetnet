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


CLASSIFICATION_CONFIGS = [
    "jetnet.torchvision.RESNET18_IMAGENET",
    "jetnet.torchvision.RESNET18_IMAGENET_TRT",
    "jetnet.torchvision.RESNET18_IMAGENET_TRT_FP16",
    "jetnet.torchvision.RESNET18_IMAGENET_TRT_INT8",
    "jetnet.torchvision.RESNET34_IMAGENET", 
    "jetnet.torchvision.RESNET34_IMAGENET_TRT", 
    "jetnet.torchvision.RESNET34_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.RESNET34_IMAGENET_TRT_INT8",
    "jetnet.torchvision.RESNET50_IMAGENET", 
    "jetnet.torchvision.RESNET50_IMAGENET_TRT", 
    "jetnet.torchvision.RESNET50_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.RESNET50_IMAGENET_TRT_INT8",
    "jetnet.torchvision.RESNET101_IMAGENET", 
    "jetnet.torchvision.RESNET101_IMAGENET_TRT", 
    "jetnet.torchvision.RESNET101_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.RESNET101_IMAGENET_TRT_INT8",
    "jetnet.torchvision.RESNET152_IMAGENET", 
    "jetnet.torchvision.RESNET152_IMAGENET_TRT", 
    "jetnet.torchvision.RESNET152_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.RESNET152_IMAGENET_TRT_INT8",
    "jetnet.torchvision.DENSENET121_IMAGENET", 
    "jetnet.torchvision.DENSENET121_IMAGENET_TRT", 
    "jetnet.torchvision.DENSENET121_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.DENSENET121_IMAGENET_TRT_INT8",
    "jetnet.torchvision.DENSENET161_IMAGENET", 
    "jetnet.torchvision.DENSENET161_IMAGENET_TRT", 
    "jetnet.torchvision.DENSENET161_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.DENSENET161_IMAGENET_TRT_INT8",
    "jetnet.torchvision.DENSENET169_IMAGENET", 
    "jetnet.torchvision.DENSENET169_IMAGENET_TRT", 
    "jetnet.torchvision.DENSENET169_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.DENSENET169_IMAGENET_TRT_INT8",
    "jetnet.torchvision.DENSENET201_IMAGENET", 
    "jetnet.torchvision.DENSENET201_IMAGENET_TRT", 
    "jetnet.torchvision.DENSENET201_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.DENSENET201_IMAGENET_TRT_INT8",
    "jetnet.torchvision.MOBILENET_V2_IMAGENET", 
    "jetnet.torchvision.MOBILENET_V2_IMAGENET_TRT", 
    "jetnet.torchvision.MOBILENET_V2_IMAGENET_TRT_FP16", 
    "jetnet.torchvision.MOBILENET_V2_IMAGENET_TRT_INT8"
]

DETECTION_CONFIGS = [
    "jetnet.yolox.YOLOX_L", 
    "jetnet.yolox.YOLOX_L_TRT", 
    "jetnet.yolox.YOLOX_L_TRT_FP16", 
    "jetnet.yolox.YOLOX_L_TRT_INT8",
    "jetnet.yolox.YOLOX_M", 
    "jetnet.yolox.YOLOX_M_TRT", 
    "jetnet.yolox.YOLOX_M_TRT_FP16", 
    "jetnet.yolox.YOLOX_M_TRT_INT8",
    "jetnet.yolox.YOLOX_S", 
    "jetnet.yolox.YOLOX_S_TRT", 
    "jetnet.yolox.YOLOX_S_TRT_FP16", 
    "jetnet.yolox.YOLOX_S_TRT_INT8",
    "jetnet.yolox.YOLOX_X", 
    "jetnet.yolox.YOLOX_X_TRT", 
    "jetnet.yolox.YOLOX_X_TRT_FP16", 
    "jetnet.yolox.YOLOX_X_TRT_INT8",
    "jetnet.yolox.YOLOX_TINY", 
    "jetnet.yolox.YOLOX_TINY_TRT", 
    "jetnet.yolox.YOLOX_TINY_TRT_FP16", 
    "jetnet.yolox.YOLOX_TINY_TRT_INT8",
    "jetnet.yolox.YOLOX_NANO", 
    "jetnet.yolox.YOLOX_NANO_TRT", 
    "jetnet.yolox.YOLOX_NANO_TRT_FP16", 
    "jetnet.yolox.YOLOX_NANO_TRT_INT8"
]

POSE_CONFIGS = [
    "jetnet.trt_pose.RESNET18_BODY_224X224", 
    "jetnet.trt_pose.RESNET18_BODY_224X224_TRT", 
    "jetnet.trt_pose.RESNET18_BODY_224X224_TRT_FP16", 
    "jetnet.trt_pose.RESNET18_BODY_224X224_TRT_INT8",
    "jetnet.trt_pose.DENSENET121_BODY_256X256", 
    "jetnet.trt_pose.DENSENET121_BODY_256X256_TRT", 
    "jetnet.trt_pose.DENSENET121_BODY_256X256_TRT_FP16", 
    "jetnet.trt_pose.DENSENET121_BODY_256X256_TRT_INT8"
]

TEXT_DETECTION_CONFIGS = [
    "jetnet.easyocr.EASYOCR_EN",
    "jetnet.easyocr.EASYOCR_EN_TRT",
    "jetnet.easyocr.EASYOCR_EN_TRT_FP16",
    "jetnet.easyocr.EASYOCR_EN_TRT_INT8_FP16",
    "jetnet.mmocr.MMOCR_DB_R18_CRNN",
    "jetnet.mmocr.MMOCR_DB_R18_CRNN_TRT",
    "jetnet.mmocr.MMOCR_DB_R18_CRNN_TRT_FP16",
    "jetnet.mmocr.MMOCR_DB_R18_CRNN_TRT_INT8_FP16",
    "jetnet.mmocr.MMOCR_DB_R18_ROBUSTSCANNER",
    "jetnet.mmocr.MMOCR_DB_R18_ROBUSTSCANNER_TRT",
    "jetnet.mmocr.MMOCR_DB_R18_ROBUSTSCANNER_TRT_FP16",
    "jetnet.mmocr.MMOCR_DB_R18_ROBUSTSCANNER_TRT_INT8_FP16"
]

IMAGE_DATASET_CONFIGS = [
    "jetnet.coco.COCO2017_VAL_IMAGES",
    "jetnet.coco.COCO2017_TRAIN_IMAGES",
    "jetnet.coco.COCO2017_TEST_IMAGES",
    "jetnet.textocr.TEXTOCR_TEST_IMAGES",
    "jetnet.textocr.TEXTOCR_TRAIN_IMAGES"
]

ALL_CONFIGS = \
    CLASSIFICATION_CONFIGS + \
    DETECTION_CONFIGS + \
    POSE_CONFIGS + \
    TEXT_DETECTION_CONFIGS + \
    IMAGE_DATASET_CONFIGS