import jetnet.coco
from jetnet.trt_pose.trt_pose_model_config import TRTPoseModelConfig
from jetnet.trt_pose.trt_pose_trt_model_config import TRTPoseTRTModelConfig


HAND_KEYPOINTS = [
    "wrist",
    "thumb_1", 
    "thumb_2", 
    "thumb_3", 
    "thumb_4", 
    "index_finger_1", 
    "index_finger_2", 
    "index_finger_3", 
    "index_finger_4", 
    "middle_finger_1", 
    "middle_finger_2", 
    "middle_finger_3", 
    "middle_finger_4", 
    "ring_finger_1", 
    "ring_finger_2", 
    "ring_finger_3", 
    "ring_finger_4", 
    "pinky_finger_1", 
    "pinky_finger_2", 
    "pinky_finger_3", 
    "pinky_finger_4"
]


HAND_SKELETON = [
    [0, 4], 
    [0, 8], 
    [0, 12], 
    [0, 16], 
    [0, 20], 
    [1, 2], 
    [2, 3], 
    [3, 4], 
    [5, 6], 
    [6, 7], 
    [7, 8], 
    [9, 10], 
    [10, 11], 
    [11, 12], 
    [13, 14], 
    [14, 15], 
    [15, 16], 
    [17, 18], 
    [18, 19], 
    [19, 20]
]

def _create_cfgs(_base_alias, _input_size, _model_name, _weights_url):
    _cfg = TRTPoseModelConfig(
        model=_model_name,
        input_size=_input_size,
        keypoints=HAND_KEYPOINTS,
        skeleton=HAND_SKELETON,
        weights_url=_weights_url,
        weights_path=f"data/trt_pose/{_base_alias}.pth"
    )
    _cfg_trt = TRTPoseTRTModelConfig(
        model=_cfg,
        int8_calib_cache=f"data/trt_pose/{_base_alias}_calib",
        engine_cache=f"data/trt_pose/{_base_alias}_trt.pth",
        int8_num_calib=512,
        int8_calib_dataset=jetnet.coco.COCO2017_VAL_IMAGES
    )
    _cfg_trt_fp16 = _cfg_trt.copy(update={"fp16_mode": True, "engine_cache": f"data/trt_pose/{_base_alias}_trt_fp16.pth"})
    _cfg_trt_int8 = _cfg_trt.copy(update={"int8_mode": True, "engine_cache": f"data/trt_pose/{_base_alias}_trt_int8.pth"})

    return _cfg, _cfg_trt, _cfg_trt_fp16, _cfg_trt_int8


RESNET18_HAND_224X224, RESNET18_HAND_224X224_TRT, RESNET18_HAND_224X224_TRT_FP16, RESNET18_HAND_224X224_TRT_INT8 = \
    _create_cfgs(
        "resnet18_hand_224x224", 
        (224, 224), 
        "resnet18_baseline_att", 
        "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/hand_pose_resnet18_att_244_244.pth"
    )