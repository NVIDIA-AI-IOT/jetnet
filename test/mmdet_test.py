import pytest
from jetnet.mmdet.mmdet import (
    MMDet
)


def test_pretrained_detector_build():

    config = MMDet(
        config="$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
        weights="data/mmdet/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        weights_url="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
    )

    detector = config.build()

    assert 'backbone' in detector.module_names()