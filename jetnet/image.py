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


from abc import ABC, abstractmethod

from typing import Optional

from jetnet.dataset import Dataset
from jetnet.utils import parent_dir, unzip, download
from jetnet.config import Config

import base64
import io
import PIL.Image
import os
import glob
import shutil
import tempfile
from PIL.Image import Image


def read_image(path: str):
    return PIL.Image.open(path).convert("RGB")


class ImageDataset(Dataset[Image]):
    pass


class _ImageFolder(ImageDataset):

    def __init__(self, path, recursive: bool = False):
        self.path = path
        self.recursive = recursive
        prefix = "**" if self.recursive else "*"
        image_paths = glob.glob(os.path.join(self.path, f"{prefix}.jpg"))
        image_paths += glob.glob(os.path.join(self.path, f"{prefix}.png"))
        image_paths += glob.glob(os.path.join(self.path, f"{prefix}.jpeg"))
        self._image_paths = image_paths

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Image:
        return read_image(self._image_paths[index])


class ImageFolder(Config[_ImageFolder]):

    path: str
    recursive: bool = False

    def build(self):
        return _ImageFolder(self.path, self.recursive)


class RemoteImageFolder(ImageFolder):

    zip_url: str
    zip_folder: str
    zip_file: str

    def build(self):

        zip_url = self.zip_url
        zip_file = self.zip_file
        zip_folder = self.zip_folder
        path = self.path

        if zip_file is None:
            zip_file = os.path.join(tempfile.mkdtemp(), "images.zip")

        if not os.path.exists(path):

            if not os.path.exists(parent_dir(zip_file)):
                os.makedirs(parent_dir(zip_file))

            if not os.path.exists(zip_file):
                tmpf = tempfile.mktemp()
                download(zip_url, tmpf)
                shutil.move(tmpf, zip_file)

            tmp = tempfile.mkdtemp()
            unzip(zip_file, tmp, pattern=os.path.join(zip_folder, "*"))

            path = os.path.abspath(path)

            if not os.path.exists(parent_dir(path)):
                os.makedirs(parent_dir(path))

            shutil.move(os.path.join(tmp, zip_folder), path)

        return super().build()