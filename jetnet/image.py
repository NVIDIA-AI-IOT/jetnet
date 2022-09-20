from abc import ABC, abstractmethod

from typing import Optional

from jetnet.config import Config
from jetnet.dataset import Dataset
from jetnet.utils import parent_dir, unzip, download

import base64
import io
import PIL.Image
import os
import glob
import shutil
import tempfile


class Image:

    def __init__(self, value: PIL.Image.Image):
        self._value = value

    def pil(self) -> PIL.Image.Image:
        return self._value

    def jpeg(self) -> bytes:
        with io.BytesIO() as f:
            self.pil().save(f, format="JPEG")
            return f.getvalue()

    @classmethod
    def from_jpeg(cls, data: bytes):
        return Image(PIL.Image.open(io.BytesIO(data)))

    @classmethod
    def from_file(cls, path: str):
        return Image(PIL.Image.open(path))


def read_image(path: str):
    return Image.from_file(path)


class ImageDataset(Dataset[Image]):
    pass


class ImageFolder(ImageDataset):

    def __init__(self, path: str, recursive: bool = False):
        self.path = path
        self.recursive = recursive
        prefix = "**" if recursive else "*"
        image_paths = glob.glob(os.path.join(path, f"{prefix}.jpg"))
        image_paths += glob.glob(os.path.join(path, f"{prefix}.png"))
        image_paths += glob.glob(os.path.join(path, f"{prefix}.jpeg"))
        self._image_paths = image_paths

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Image:
        return read_image(self._image_paths[index])


class ImageDatasetConfig(Config[ImageDataset]):
    pass


class ImageFolderConfig(ImageDatasetConfig):
    path: str
    recursive: bool = False

    def build(self):
        from jetnet.image import ImageFolder
        
        return ImageFolder(self.path, self.recursive)


class RemoteImageFolderConfig(ImageDatasetConfig):

    image_folder: ImageFolderConfig
    zip_url: str
    zip_folder: str
    zip_file: str

    def build(self):
        import os
        import glob
        import shutil
        import tempfile
        from jetnet.utils import download, unzip, parent_dir

        zip_url = self.zip_url
        zip_file = self.zip_file
        zip_folder = self.zip_folder
        path = self.image_folder.path

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

        return self.image_folder.build()