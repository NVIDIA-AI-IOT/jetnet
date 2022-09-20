import jetnet.coco
from jetnet.trt_pose.trt_pose import TRTPose, TRTPoseTRT

BODY_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
]

BODY_SKELETON = [
    [15, 13], 
    [13, 11], 
    [16, 14], 
    [14, 12], 
    [11, 12], 
    [5, 7], 
    [6, 8], 
    [7, 9], 
    [8, 10], 
    [1, 2], 
    [0, 1], 
    [0, 2], 
    [1, 3], 
    [2, 4], 
    [3, 5], 
    [4, 6], 
    [17, 0], 
    [17, 5], 
    [17, 6], 
    [17, 11], 
    [17, 12]
]



def _create_cfgs(_base_alias, _input_size, _model_name, _weights_url):
    _cfg = TRTPose(
        model=_model_name,
        input_size=_input_size,
        keypoints=BODY_KEYPOINTS,
        skeleton=BODY_SKELETON,
        weights_url=_weights_url,
        weights_path=f"data/trt_pose/{_base_alias}.pth"
    )
    _cfg_trt = TRTPoseTRT(
        model=_cfg,
        int8_calib_cache=f"data/trt_pose/{_base_alias}_calib",
        engine_cache=f"data/trt_pose/{_base_alias}_trt.pth",
        int8_num_calib=512,
        int8_calib_dataset=jetnet.coco.COCO2017_VAL_IMAGES
    )
    _cfg_trt_fp16 = _cfg_trt.copy(update={"fp16_mode": True, "engine_cache": f"data/trt_pose/{_base_alias}_trt_fp16.pth"})
    _cfg_trt_int8 = _cfg_trt.copy(update={"int8_mode": True, "engine_cache": f"data/trt_pose/{_base_alias}_trt_int8.pth"})

    return _cfg, _cfg_trt, _cfg_trt_fp16, _cfg_trt_int8


RESNET18_BODY_224X224, RESNET18_BODY_224X224_TRT, RESNET18_BODY_224X224_TRT_FP16, RESNET18_BODY_224X224_TRT_INT8 = \
    _create_cfgs(
        "resnet18_body_224x224", 
        (224, 224), 
        "resnet18_baseline_att",
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth"
    )


DENSENET121_BODY_256X256, DENSENET121_BODY_256X256_TRT, DENSENET121_BODY_256X256_TRT_FP16, DENSENET121_BODY_256X256_TRT_INT8 = \
    _create_cfgs(
        "densenet121_body_256x256", 
        (256, 256), 
        "densenet121_baseline_att",
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/densenet121_baseline_att_256x256_B_epoch_160.pth"
    )