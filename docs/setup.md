This page details setup steps needed to start using JetNet

## Docker Setup

JetNet comes with pre-built docker containers for some system configurations.
If you have the disk space and there is an available container, this is a fast and easy option for getting started.  

To use the container, first clone the github repo

```bash
git clone https://github.com/NVIDIA-AI-IOT/jetnet
cd jetnet
```

Next, launch the docker container from inside the cloned directory

=== "Jetson (JetPack 5.0.2)"

    ```bash
    docker run \
        --network host \
        --gpus all \
        -it \
        --rm \
        --name=jetnet \
        -v $(pwd):/jetnet \
        --device /dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        jaybdub/jetnet:l4t-35.1.0 \
        /bin/bash -c "cd /jetnet && python3 setup.py develop && /bin/bash"
    ```

=== "Jetson (JetPack 5.0.1)"

    ```bash
    docker run \
        --network host \
        --gpus all \
        -it \
        --rm \
        --name=jetnet \
        -v $(pwd):/jetnet \
        --device /dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        jaybdub/jetnet:l4t-34.1.1 \
        /bin/bash -c "cd /jetnet && python3 setup.py develop && /bin/bash"
    ```

=== "Desktop (NV driver 465.19.01+)"

    ```bash
    docker run \
        --network host \
        --gpus all \
        -it \
        --rm \
        --name=jetnet \
        -v $(pwd):/jetnet \
        --device /dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        jaybdub/jetnet:x86-21.05 \
        /bin/bash -c "cd /jetnet && python3 setup.py develop && /bin/bash"
    ```

This will mount the current directory (which should be the jetnet project root) at ``/jetnet`` inside
the container.  Most data downloaded when using JetNet is stored in the ``data`` folder.  So
assuming you use JetNet command line tools from ``/jetnet`` inside the container, the data will persist
upon container restart.

> Note, this command assumes you have a USB camera at /dev/video0.  Please adjust the command accordingly.

<details>
<summary>Building docker containers</summary>

You may want to build the containers yourself, if you have additional dependencies, or need to use
a different base container. Below are the commands we use to build the pre-made containers.
Check the GitHub repo docker files for more details.


```bash
docker build -t jaybdub/jetnet:l4t-34.1.1 -f $(pwd)/docker/l4t-34.1.1/Dockerfile $(pwd)/docker/l4t-34.1.1
```

</details>

  
## Manual Setup

If there is not a container available for your platform, or you don't have the storage
space, you can set up your system natively.

- Install TensorRT, PyTorch, OpenCV and Torchvision (please refer to external instructions)
- Install miscellanerous dependencies

    ```bash
    pip3 install pydantic progressbar python3-socketio uvicorn starlette
    ```

- Install torch2trt

    ```bash
    pip3 install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@master
    ```

- Install YOLOX (required for ``jetnet.yolox``)

    ```bash
    git clone https://github.com/Megvii-BaseDetection/YOLOX
    cd YOLOX
    python3 setup.py install
    cd ..
    ```

- Install EasyOCR (required for ``jetnet.easyocr``)

    ```bash
    pip3 install git+https://github.com/JaidedAI/EasyOCR.git@v1.5.0
    ```

- Install TRTPose (required for ``jetnet.trt_pose``)

    ```bash
    pip3 install git+https://github.com/NVIDIA-AI-IOT/trt_pose.git
    ```

- Install JetNet

    ```bash
    git clone https://github.com/NVIDIA-AI-IOT/jetnet
    cd jetnet
    python3 setup.py develop
    ```

> Currently we exclude ``jetnet.mmocr`` from manual setup. For now, please
> reference the dockerfile in the GitHub repo if you wish to use these models.
