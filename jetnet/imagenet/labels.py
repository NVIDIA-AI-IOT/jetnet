from .label_map import *


IMAGENET_LABELS = [""] * len(IMAGENET_LABEL_MAP)
for k, v in IMAGENET_LABEL_MAP.items():
    IMAGENET_LABELS[k] = v