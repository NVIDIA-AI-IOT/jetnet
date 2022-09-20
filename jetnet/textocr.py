from jetnet.image import (
    RemoteImageFolderConfig,
    ImageFolderConfig
)

__all__ = [
    "textocr_test_images",
    "textocr_train_images"
]


TEXTOCR_TEST_IMAGES = RemoteImageFolderConfig(
    image_folder=ImageFolderConfig(path="data/textocr/textocr_test_images/test_images"),
    zip_url="https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
    zip_folder="test_images",
    zip_file="data/textocr/textocr_test_images/test_images.zip"
)


TEXTOCR_TRAIN_IMAGES = RemoteImageFolderConfig(
    image_folder=ImageFolderConfig(path="data/textocr/textocr_train_images/train_images"),
    zip_url="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    zip_folder="train_images",
    zip_file="data/textocr/textocr_train_images/train_val_images.zip"
)