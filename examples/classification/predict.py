from jetnet.image import read_image
from jetnet.torchvision import RESNET18_IMAGENET_TRT


model = RESNET18_IMAGENET_TRT.build()

image = read_image("assets/dog.jpg")

output = model(image)

print(output.json(indent=2))