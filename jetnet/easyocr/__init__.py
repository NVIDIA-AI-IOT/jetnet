from jetnet.easyocr.easyocr_model_config import EasyOCRModelConfig
from jetnet.easyocr.easyocr_trt_model_config import (
    EasyOCRTRTModelConfig,
    EasyOCRTRTEngineConfig,
    EASYOCR_TRT_DEFAULT_DETECTOR_CFG,
    EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG
)
import jetnet.textocr


EASYOCR_EN = EasyOCRModelConfig(
    lang_list=["en"]
)

EASYOCR_EN_TRT = EasyOCRTRTModelConfig(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "engine_cache": "data/easyocr/easyocr_en_det_trt.pth",
        "int8_calib_cache": "data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "engine_cache": "data/easyocr/easyocr_en_rec_trt.pth",
        "int8_calib_cache": "data/easyocr/easyocr_en_rec_calib"
    })
)

EASYOCR_EN_TRT_FP16 = EasyOCRTRTModelConfig(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_det_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_rec_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_rec_calib"
    })
)

EASYOCR_EN_TRT_INT8_FP16 = EasyOCRTRTModelConfig(
    model=EASYOCR_EN,
    int8_calib_dataset=jetnet.textocr.TEXTOCR_TEST_IMAGES,
    detector_config=EASYOCR_TRT_DEFAULT_DETECTOR_CFG.copy(update={
        "int8_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_det_trt_int8.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_det_calib"
    }),
    recognizer_config=EASYOCR_TRT_DEFAULT_RECOGNIZER_CFG.copy(update={
        "fp16_mode": True,
        "engine_cache": f"data/easyocr/easyocr_en_rec_trt_fp16.pth",
        "int8_calib_cache": f"data/easyocr/easyocr_en_rec_calib"
    })
)
