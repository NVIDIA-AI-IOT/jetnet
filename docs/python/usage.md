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
