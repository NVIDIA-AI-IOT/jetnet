import pytest
from jetnet.mmdet import (
    MMDet,
    MASK_RCNN_R50_FPN_1X_COCO_TRT_FP16
)
from mmdet.apis import inference_detector
from jetnet.image import read_image

def test_pretrained_detector_build():

    config = MMDet(
        config="$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
        weights="data/mmdet/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        weights_url="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
    )

    detector = config.build()

    assert 'backbone' in detector.module_names()


def test_mask_rcnn_r50():

    model = MASK_RCNN_R50_FPN_1X_COCO_TRT_FP16.build()

    image = read_image("assets/person.jpg")

    output = model(image)

    assert len(output.detections) > 0