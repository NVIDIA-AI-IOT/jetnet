Pick a pre-defined [model](models.md) and use it with these tools.
For this example, we'll use the ``jetnet.trt_pose.RESNET18_BODY_224X224_TRT_FP16`` model.

> You can also define your our own config and use it with these
> tools as long as it can be imported in Python.

### Build

``jetnet build`` is a convenience tool that simply imports the model and calls ``model.build()``.  This is useful for testing if a model builds and for generating cached data before using the model elsewhere.

To use it, call ``jetnet build <model>``.  For example,


=== "Classification"

    ```bash
    jetnet build jetnet.torchvision.RESNET18_IMAGENET_TRT_FP16
    ```

=== "Detection"

    ```bash
    jetnet build jetnet.yolox.YOLOX_NANO_TRT_FP16
    ```

=== "Pose"

    ```bash
    jetnet build jetnet.trt_pose.RESNET18_BODY_224X224_TRT_FP16
    ```

=== "Text Detection"

    ```bash
    jetnet build jetnet.easyocr.EASYOCR_EN_TRT_FP16
    ```

### Profile

``jetnet profile`` profiles a model on *real* data.  It measures the model throughput,
as well as other task specific statistics like the average number of objects per image.  This
is handy, especially for models that may have data-dependent runtime.

To use it, call ``jetnet build <model> <dataset>``.  For example,


=== "Classification"

    ```bash
    jetnet profile jetnet.torchvision.RESNET18_IMAGENET_TRT_FP16 jetnet.coco.COCO2017_VAL_IMAGES
    ```

    example output:
    
    ```json
    {
        "fps": 159.50995530176425,
        "avg_image_area": 286686.08
    }
    ```

=== "Detection"

    ```bash
    jetnet profile jetnet.yolox.YOLOX_NANO_TRT_FP16 jetnet.coco.COCO2017_VAL_IMAGES
    ```

    example output:
    
    ```json
    {
        "fps": 161.64499986521764,
        "avg_image_area": 286686.08,
        "avg_num_detections": 3.86
    }
    ```

=== "Pose"

    ```bash
    jetnet profile jetnet.trt_pose.RESNET18_BODY_224X224_TRT_FP16 jetnet.coco.COCO2017_VAL_IMAGES
    ```

    example output:

    ```json
    {
        "fps": 103.2494480449782,
        "avg_image_area": 286686.08,
        "avg_num_poses": 1.98,
        "avg_num_keypoints": 16.32
    }
    ```

=== "Text Detection"

    ```bash
    jetnet profile jetnet.easyocr.EASYOCR_EN_TRT_FP16 jetnet.textocr.TEXTOCR_TEST_IMAGES
    ```
    
    example output:

    ```json
    {
        "fps": 13.334012937781655,
        "avg_image_area": 768962.56,
        "avg_num_detections": 10.48,
        "avg_num_characters": 66.46
    }
    ```


### Demo

``jetnet demo`` peforms inference on live camera images and displays predictions in your web browser.
This is especially handy when you're operating on a headless machine.

With a USB camera attached, call ``jetnet demo <model>``.  For example,

=== "Classification"

    ```bash
    jetnet demo jetnet.torchvision.RESNET18_IMAGENET_TRT_FP16
    ```

    Once the demo is running, navigate to ``http://<ip>:8000`` in your web browser to view the predictions.

    <img src="https://user-images.githubusercontent.com/4212806/191136464-8f3c05fc-9e70-4678-9402-6d4d8232661b.gif">

=== "Detection"

    ```bash
    jetnet demo jetnet.yolox.YOLOX_NANO_TRT_FP16
    ```

    Once the demo is running, navigate to ``http://<ip>:8000`` in your web browser to view the predictions.

    <img src="https://user-images.githubusercontent.com/4212806/191136616-06ce3640-7e35-45a3-8b2e-7f7a5b9b7f28.gif">

=== "Pose"

    ```bash
    jetnet demo jetnet.trt_pose.RESNET18_BODY_224X224_TRT_FP16
    ```
    
    Once the demo is running, navigate to ``http://<ip>:8000`` in your web browser to view the predictions.

    <img src="https://user-images.githubusercontent.com/4212806/191136450-4b2d55c1-c3c7-47d6-996e-11c62448747b.gif">

=== "Text Detection"

    ```bash
    jetnet demo jetnet.easyocr.EASYOCR_EN_TRT_FP16
    ```
    
    Once the demo is running, navigate to ``http://<ip>:8000`` in your web browser to view the predictions.
    
    <img src="https://user-images.githubusercontent.com/4212806/191136896-e42ab4d9-3a2f-4553-a1c7-49c59fc7e7a2.gif">
