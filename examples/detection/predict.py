from jetnet.image import read_image
from jetnet.yolox import YOLOX_TINY_TRT_FP16


model = YOLOX_TINY_TRT_FP16.build()

image = read_image("assets/dog.jpg")

output = model(image)

print(output.json(indent=2))