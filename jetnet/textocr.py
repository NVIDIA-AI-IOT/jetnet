from jetnet.image import (
    RemoteImageFolder
)

__all__ = [
    "TEXTOCR_TEST_IMAGES",
    "TEXTOCR_TRAIN_IMAGES"
]


TEXTOCR_TEST_IMAGES = RemoteImageFolder(
    path="data/textocr/textocr_test_images/test_images",
    zip_url="https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
    zip_folder="test_images",
    zip_file="data/textocr/textocr_test_images/test_images.zip"
)


TEXTOCR_TRAIN_IMAGES = RemoteImageFolder(
    path="data/textocr/textocr_train_images/train_images",
    zip_url="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    zip_folder="train_images",
    zip_file="data/textocr/textocr_train_images/train_val_images.zip"
)