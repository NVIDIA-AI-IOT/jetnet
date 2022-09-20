# JetNet

<img src="https://user-images.githubusercontent.com/4212806/191136464-8f3c05fc-9e70-4678-9402-6d4d8232661b.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136616-06ce3640-7e35-45a3-8b2e-7f7a5b9b7f28.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136450-4b2d55c1-c3c7-47d6-996e-11c62448747b.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136896-e42ab4d9-3a2f-4553-a1c7-49c59fc7e7a2.gif" height="25%" width="25%"/>

JetNet is a collection of **models**, **datasets**, and
**tools** that make it easy to explore neural networks on NVIDIA Jetson (and desktop too!). It can easily be used and extended with **Python**.  

Check out the [documentation](#) to learn more and get started!

### It's easy to use

JetNet comes with tools that allow you to easily build, profile and demo models.  This helps you easily try out models to see what is right for your application.  

```bash
jetnet demo jetnet.trt_pose.RESNET18_HAND_224X224_TRT_FP16
```

<img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif"/>


### It's implementation agnostic

JetNet has well defined interfaces for tasks like classification, detection, pose estimation, and text detection.  This means models have a familiar interface, regardless of which framework they are implemented in.  As a user, this lets you easily use a variety of models without re-learning
a new interface for each one. 

```python3
class PoseModel:

    def get_keypoints(self) -> Sequence[str]:
        raise NotImplementedError

    def get_skeleton(self) -> Sequence[Tuple[int, int]]:
        raise NotImplementedError

    def __call__(self, index: Image) -> PoseSet:
        raise NotImplementedError
```

### It's highly reproducible and configurable

JetNet uses well-defined configurations to explicitly describe all the steps needed to automatically re-produce a model.  This includes steps like downloading weights, downloading calibration data and optimizing with TensorRT, that often aren't captured in open-source model definitons.  These configurations are defined with ``pydantic`` using JSON serializable so they can be easily validated, modified, exported, and re-used.

For example, the following models, which include TensorRT optimization can be re-created with a single line

```python
from jetnet.yolox import YOLOX_NANO_TRT_FP16

model = YOLOX_NANO_TRT_FP16.build()
```

### It's easy to set up

JetNet comes with pre-built docker containers for Jetson and Desktop.
In case these don't work for you, manual setup instructions are provided.
Check out the documentation for details.

### It's extensible

JetNet is written with Python so that it is easy
to extend.  If you want to use the JetNet tools with a different model, or are
considering contributing to the project to help other developers easily use your model, all you need to do is implement one of the JetNet interfaces and define a config for
re-creating the model.  

For example, here's a dummy example for a classifier that returns
a constant class label.

```python
class CatDogModel(ClassificationModel):
    
    def __init__(self, default_label: str):
        self.default_label = default_label

    def get_labels(self) -> Sequence[str]:
        return ["cat", "dog"]

    def __call__(self, x: Image) -> Classification:
        return Classification(
            index=self.get_labels().index(self.default_label), 
            label=self.default_label
        )

class CatDogModelConfig(ClassificationModelConfig):
    
    default_label: str

    def build(self) -> CatDogModel:
        return CatDogModel(self.default_label)

ALWAYS_DOG = CatDogModelConfig(default_label="dog")
ALWAYS_CAT = CatDogModelConfig(default_label="cat")
```

We can then use the model with JetNet tools.

```bash
jetnet demo cat_dog.ALWAYS_CAT
```


## Get Started!

Head on over the [documentation](#) to learn more and get started!
