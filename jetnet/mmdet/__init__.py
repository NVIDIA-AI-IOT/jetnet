from .mmdet import MMDet
from jetnet.torch2trt import (
    Torch2trtModel,
    Torch2trtEngineConfig,
    Torch2trtInputSpec
)

MASK_RCNN_R50_FPN_1X_COCO = MMDet(
    config="$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
    weights="data/mmdet/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
    weights_url="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
)

MINW, OPTW, MAXW = 320, 1344, 1344
MINH, OPTH, MAXH = 320, 800, 1344

MASK_RCNN_R50_FPN_1X_COCO_TRT_FP16 = Torch2trtModel(
    model=MASK_RCNN_R50_FPN_1X_COCO,
    engine_configs={
        "backbone": Torch2trtEngineConfig(
            inputs=[Torch2trtInputSpec(min_shape=[1, 3, MINH, MINW], opt_shape=[1, 3, OPTH, OPTW], max_shape=[1, 3, MAXH, MAXW])],
            fp16_mode=True,
            use_onnx=True
        ),
        "neck": Torch2trtEngineConfig(
            inputs=[[
                Torch2trtInputSpec(min_shape=[1, 256, MINH//4, MINW//4], opt_shape=[1, 256, OPTH//4, OPTW//4], max_shape=[1, 256, MAXH//4, MAXW//4]),
                Torch2trtInputSpec(min_shape=[1, 512, MINH//8, MINW//8], opt_shape=[1, 512, OPTH//8, OPTW//8], max_shape=[1, 512, MAXH//8, MAXW//8]),
                Torch2trtInputSpec(min_shape=[1, 1024, MINH//16, MINW//16], opt_shape=[1, 1024, OPTH//16, OPTW//16], max_shape=[1, 1024, MAXH//16, MAXW//16]),
                Torch2trtInputSpec(min_shape=[1, 2048, MINH//32, MINW//32], opt_shape=[1, 2048, OPTH//32, OPTW//32], max_shape=[1, 2048, MAXH//32, MAXW//32])
            ]],
            fp16_mode=True,
            use_onnx=True
        ),
        "bbox": Torch2trtEngineConfig(
            inputs=[
                Torch2trtInputSpec(min_shape=[1, 256, 7, 7], opt_shape=[8, 256, 7, 7], max_shape=[1000, 256, 7, 7]),
            ],
            fp16_mode=True,
            use_onnx=True
        ),
        "mask": Torch2trtEngineConfig(
            inputs=[
                Torch2trtInputSpec(min_shape=[1, 256, 14, 14], opt_shape=[8, 256, 14, 14], max_shape=[100, 256, 14, 14]),
            ],
            fp16_mode=True,
            use_onnx=True
        )
    }
)

MASK_RCNN_R50_FPN_1X_COCO_TRT = MASK_RCNN_R50_FPN_1X_COCO_TRT_FP16.copy(deep=True)
MASK_RCNN_R50_FPN_1X_COCO_TRT.engine_configs['backbone'].fp16_mode = False
MASK_RCNN_R50_FPN_1X_COCO_TRT.engine_configs['neck'].fp16_mode = False
MASK_RCNN_R50_FPN_1X_COCO_TRT.engine_configs['bbox'].fp16_mode = False
MASK_RCNN_R50_FPN_1X_COCO_TRT.engine_configs['mask'].fp16_mode = False