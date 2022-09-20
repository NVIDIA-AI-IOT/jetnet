from jetnet.image import read_image
from jetnet.trt_pose import RESNET18_BODY_224X224_TRT_FP16


model = RESNET18_BODY_224X224_TRT_FP16.build()

image = read_image("assets/person.jpg")

output = model(image)

print(output.json(indent=2))