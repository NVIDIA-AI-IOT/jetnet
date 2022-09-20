from jetnet.image import (
    RemoteImageFolderConfig,
    ImageFolderConfig
)

__all__ = [
    "COCO2017_VAL_IMAGES",
    "coco2017_train_images",
    "coco2017_test_images"
]


COCO2017_VAL_IMAGES = RemoteImageFolderConfig(
    image_folder=ImageFolderConfig(path="data/coco/val2017"),
    zip_url="http://images.cocodataset.org/zips/val2017.zip",
    zip_file="data/coco/val2017.zip",
    zip_folder="val2017"
)

COCO2017_TRAIN_IMAGES = RemoteImageFolderConfig(
    image_folder=ImageFolderConfig(path="data/coco/train2017"),
    zip_url="http://images.cocodataset.org/zips/train2017.zip",
    zip_file="data/datasets/coco/train2017.zip",
    zip_folder="train2017"
)

COCO2017_TEST_IMAGES = RemoteImageFolderConfig(
    image_folder=ImageFolderConfig(path="data/coco/test2017"),
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
