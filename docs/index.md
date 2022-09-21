<img src="https://user-images.githubusercontent.com/4212806/191136464-8f3c05fc-9e70-4678-9402-6d4d8232661b.gif" style="max-height:160px;"/>
<img src="https://user-images.githubusercontent.com/4212806/191136616-06ce3640-7e35-45a3-8b2e-7f7a5b9b7f28.gif" style="max-height:160px;"/>
<img src="https://user-images.githubusercontent.com/4212806/191136450-4b2d55c1-c3c7-47d6-996e-11c62448747b.gif" style="max-height:160px;"/>
<img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif" style="max-height:160px;"/>
<img src="https://user-images.githubusercontent.com/4212806/191136896-e42ab4d9-3a2f-4553-a1c7-49c59fc7e7a2.gif" style="max-height:160px;"/>

JetNet is a collection of [models](models.md), [datasets](datasets.md), and
[tools](tools.md) that make it easy to explore neural networks on NVIDIA Jetson (and desktop too!). It can easily be used and extended with [Python](python/usage.md).  


### It easy to use
<!-- 
<div style="display: inline-block"> -->

<!-- <img src="assets/dog.jpg" style="max-width:256px;" align="right"> -->

JetNet comes with tools that allow you to easily <a href="tools/#build">build</a> , <a href="tools/#profile">profile</a> and <a href="tools/#demo">demo</a> models.  This helps you easily try out models to see what is right for your application.  

For example, here is how you would run a live web demo for different tasks

=== "Classification"

    ```bash
    jetnet demo jetnet.torchvision.RESNET18_IMAGENET_TRT_FP16
    ```

    and then open your browser to ``<ip_address>:8000`` to view the detections:

    <img src="https://user-images.githubusercontent.com/4212806/191136464-8f3c05fc-9e70-4678-9402-6d4d8232661b.gif">

=== "Detection"

    ```bash
    jetnet demo jetnet.yolox.YOLOX_NANO_TRT_FP16
    ```

    and then open your browser to ``<ip_address>:8000`` to view the detections:

    <img src="https://user-images.githubusercontent.com/4212806/191136616-06ce3640-7e35-45a3-8b2e-7f7a5b9b7f28.gif">

=== "Pose"

    ```bash
    jetnet demo jetnet.trt_pose.RESNET18_HAND_224X224_TRT_FP16
    ```

    and then open your browser to ``<ip_address>:8000`` to view the detections:
    
    <img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif">

=== "Text Detection"

    ```bash
    jetnet demo jetnet.easyocr.EASYOCR_EN_TRT_FP16
    ```
    
    and then open your browser to ``<ip_address>:8000`` to view the detections:

    <img src="https://user-images.githubusercontent.com/4212806/191136896-e42ab4d9-3a2f-4553-a1c7-49c59fc7e7a2.gif">


### It's implementation agnostic

JetNet has well defined interfaces for tasks like [classification](python/reference/#classificationmodel), [detection](python/reference/#detectionmodel), [pose estimation](python/reference/#posemodel), and [text detection](python/reference/#textdetectionmodel).  This means models have a familiar interface, regardless of which framework they are implemented in.  As a user, this lets you easily use a variety of models without re-learning
a new interface for each one. 

=== "Classification"

    ```python3
    class ClassificationModel:

        def init(self):
            pass

        def get_labels(self) -> Sequence[str]:
            raise NotImplementedError
        
        def __call__(self, x: Image) -> Classification:
            raise NotImplementedError
    ```

=== "Detection"

    ```python3
    class DetectionModel:

        def init(self):
            pass

        def get_labels(self) -> Sequence[str]:
            raise NotImplementedError
        
        def __call__(self, x: Image) -> DetectionSet:
            raise NotImplementedError
    ```

=== "Pose"

    ```python3
    class PoseModel:

        def init(self):
            pass

        def get_keypoints(self) -> Sequence[str]:
            raise NotImplementedError

        def get_skeleton(self) -> Sequence[Tuple[int, int]]:
            raise NotImplementedError

        def __call__(self, x: Image) -> PoseSet:
            raise NotImplementedError
    ```

=== "Text Detection"

    ```python3
    class TextDetectionModel:

        def init(self):
            pass
            
        def __call__(self, x: Image) -> TextDetectionSet:
            raise NotImplementedError
    ```

### It's highly reproducible and configurable

JetNet models are defined as ``pydantic`` types, which means they they can be easily validated, modified, and exported to JSON.  The models include an ``init`` function which is used to perform all steps necessary to prepare the model for execution, like downloading weights, downloading calibration data and optimizing with TensorRT.  

For example, the following models, which include TensorRT optimization can be re-created with a single line

=== "Classification"

    ```python
    from jetnet.torchvision import RESNET18_IMAGENET_TRT_FP16

    model = RESNET18_IMAGENET_TRT_FP16.build()
    ```

=== "Detection"

    ```python
    from jetnet.yolox import YOLOX_NANO_TRT_FP16

    model = YOLOX_NANO_TRT_FP16.build()
    ```

=== "Pose"

    ```python
    from jetnet.trt_pose import RESNET18_BODY_224X224_TRT_FP16

    model = RESNET18_BODY_224X224_TRT_FP16.build()
    ```

=== "Text Detection"

    ```python
    from jetnet.easyocr import EASYOCR_EN_TRT_FP16

    model = EASYOCR_EN_TRT_FP16.build()
    ```

### It's easy to set up

<!-- <div style="display: inline-block"> -->
    
<!-- <img src="assets/dog.jpg" style="max-width:256px;" align="left"> -->
    
JetNet comes with pre-built docker containers for Jetson and Desktop.
In case these don't work for you, manual setup instructions are provided.
Check out the <a href="setup">Setup</a> page for details.

<!-- </div> -->


<!-- </div> -->

### It's extensible

<!-- <div style="display: inline-block"> -->

<!-- <img src="assets/dog.jpg" style="max-width:256px;" align="left"> -->

JetNet is written with <a href="python/usage">Python</a> so that it is easy
to extend.  If you want to use the JetNet tools with a different model, or are
considering contributing to the project to help other developers easily use your model, all you need to do is implement one of the JetNet [interfaces](python/reference/#abstract-types).

For example, here's how we might define a new classification model

=== "Definition (``cat_dog.py``)"

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

    CATDOG_SMALL = CatDogModel(num_layers=10)
    CATDOG_BIG = CatDogModel(num_layers=50)
    ```

We can then use the model with JetNet tools.

```bash
jetnet demo cat_dog.CATDOG_SMALL
```

## Get Started!

Head on over the [Setup](setup) to configure your system to run JetNet.

Please note, if a task isn't supported that you would like to see in JetNet, let us know on GitHub.  You can open an issue, discussion or even a pull-request to get things started.
We welcome all feedback!

<!-- </div> -->