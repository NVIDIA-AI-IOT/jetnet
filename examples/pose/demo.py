from jetnet.demo.pose_demo import PoseDemoConfig
from jetnet.trt_pose import RESNET18_BODY_224X224_TRT_FP16

config = PoseDemoConfig(
    model_config=RESNET18_BODY_224X224_TRT_FP16
)

demo = config.build()

demo.run()