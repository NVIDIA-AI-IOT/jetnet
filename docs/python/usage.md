## Model usage

### Build a model

To build a model, call the following

```python3
from jetnet.yolox import YOLOX_TINY_TRT_FP16

model = YOLOX_TINY_TRT_FP16.build()
```

### Perform inference

Once the model is built, you can then perform inference 

```python3
import PIL.Image

image = PIL.Image.open("assets/person.jpg")

output = model(image)

print(output.json(indent=2))
```

<details>
    <summary>Output</summary>

```json
{
"detections": [
    {
    "boundary": {
        "points": [
        {
            "x": 312,
            "y": 262
        },
        {
            "x": 667,
            "y": 262
        },
        {
            "x": 667,
            "y": 1304
        },
        {
            "x": 312,
            "y": 1304
        }
        ]
    },
    "classification": {
        "index": 0,
        "label": "person",
        "score": 0.9122651219367981
    }
    }
]
}
```
</details>

### Customize a model

You can customize the model by copying and modifying it before building

```python3
model = YOLOX_TINY_TRT_FP16.copy(deep=True)

model.model.input_size = (1280, 736)
model.engine_cache = "data/custom_model.pth"

model = model.build()
```

### Dump a model to JSON

All models are JSON serializable, so we can view the model like this

```python3
print(model.json(indent=2))
```

<details>
<summary>Output</summary>

```json
{
  "model": {
    "exp": "yolox_tiny",
    "input_size": [
      1280,
      736
    ],
    "labels": [
      "person",
      "bicycle",
      "car",
      "motorcycle",
      "airplane",
      "bus",
      "train",
      "truck",
      "boat",
      "traffic light",
      "fire hydrant",
      "stop sign",
      "parking meter",
      "bench",
      "bird",
      "cat",
      "dog",
      "horse",
      "sheep",
      "cow",
      "elephant",
      "bear",
      "zebra",
      "giraffe",
      "backpack",
      "umbrella",
      "handbag",
      "tie",
      "suitcase",
      "frisbee",
      "skis",
      "snowboard",
      "sports ball",
      "kite",
      "baseball bat",
      "baseball glove",
      "skateboard",
      "surfboard",
      "tennis racket",
      "bottle",
      "wine glass",
      "cup",
      "fork",
      "knife",
      "spoon",
      "bowl",
      "banana",
      "apple",
      "sandwich",
      "orange",
      "broccoli",
      "carrot",
      "hot dog",
      "pizza",
      "donut",
      "cake",
      "chair",
      "couch",
      "potted plant",
      "bed",
      "dining table",
      "toilet",
      "tv",
      "laptop",
      "mouse",
      "remote",
      "keyboard",
      "cell phone",
      "microwave",
      "oven",
      "toaster",
      "sink",
      "refrigerator",
      "book",
      "clock",
      "vase",
      "scissors",
      "teddy bear",
      "hair drier",
      "toothbrush"
    ],
    "conf_thresh": 0.3,
    "nms_thresh": 0.3,
    "device": "cuda",
    "weights_path": "data/yolox/yolox_tiny.pth",
    "weights_url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
  },
  "int8_mode": false,
  "fp16_mode": true,
  "max_workspace_size": 33554432,
  "engine_cache": "data/custom_model.pth",
  "int8_calib_dataset": {
    "image_folder": {
      "path": "data/coco/val2017",
      "recursive": false
    },
    "zip_url": "http://images.cocodataset.org/zips/val2017.zip",
    "zip_folder": "val2017",
    "zip_file": "data/coco/val2017.zip"
  },
  "int8_calib_cache": "data/yolox/yolox_tiny_calib",
  "int8_num_calib": 512,
  "int8_calib_algorithm": "entropy_2"
}
```

</details>

### Define a custom model class

You can create your own model by subclassing the related base model for the task. For example,

```python
from pydantic import PrivateAttr

class CatDogModel(ClassificationModel):
    
    num_layers: int

    # private attributes can be non-JSON types, like a PyTorch module
    _torch_module = PrivateAttr()
    
    def init(self):
        # code to initialize model for execution

    def get_labels(self) -> Sequence[str]:
        return ["cat", "dog"]

    def __call__(self, x: Image) -> Classification:
        # code to classify image
```

You can then define some instances of your model

```python3
CAT_DOG_SMALL = CatDogModel(num_layers=10)
CAT_DOG_BIG = CatDogModel(num_layers=50)
```

If the model can be imported in Python, it can be used
with the command line tools.  Suppose we have our models defined in ``./cat_dog.py``,
we could use a model like this

```python3
jetnet demo cat_dog.CAT_DOG_SMALL
```

## Dataset usage

### Build a dataset

First, you build the dataset like this

```python3
from jetnet.coco import COCO2017_VAL_IMAGES

dataset = COCO2017_VAL_IMAGES.build()
```

### Get the size

Once the dataset is built, you can determine the size
of the dataset using the ``len`` method like this

```python3
size = len(dataset)
```

### Read a sample

To read a sample from the dataset, do this

```python3
image = dataset[0]
```

### Use your own images

If you have images in a folder, you can create a dataset
for them like this

```python3
CAT_DOG_IMAGES = ImageFolder(path="images")
```

Assuming this is defined in ``./cat_dog.py`` you could then
use it with the command line tools like this

```python3
jetnet profile cat_dog.CAT_DOG_SMALL cat_dog.CAT_DOG_IMAGES
```

> It's worth checking out the RemoteImageFolder, so
> you can store your images remotely, and automatically
> download it.  This will make your dataset more reproducible.


### Define a custom dataset class

If the pre-made dataset classes don't fit your use case, you
can create your own dataset class like this

```python3
from jetnet.image import ImageDataset

class CatDogImages(ImageDataset):

    def init(self):
        # code to prepare dataset for reading, ie: downloading data

    def __len__(self) -> int:
        # code to get length of dataset

    def __getitem__(self) -> Image:
        # code to read sample from dataset
```
