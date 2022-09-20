from jetnet.image import read_image
from jetnet.easyocr import EASYOCR_EN_TRT_FP16


model = EASYOCR_EN_TRT_FP16.build()

image = read_image("assets/text.jpg")

output = model(image)

print(output.json(indent=2))