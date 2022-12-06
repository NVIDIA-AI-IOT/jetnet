This page details setup steps for running multiple JetNet containers with a webcam as a shared input

## Webcam loopback setup

### V4L2 loopback device creation

To share a webcam across multiple containers, set up a v4l2loopback device.

```bash
sudo apt-get update
sudo apt-get install v4l2loopback-dkms
```

Attach your webcam to Jetson and check if it's detected on your Jetson.

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

For your webcam to be `/dev/video0`, create a v4l2loopback device with device ID `10` (incremented by 10).

```bash
sudo modprobe v4l2loopback video_nr=10 exclusive_caps=1 card_label="Webcam alias"
```

You should find `/dev/video10` created.

### Start streaming to the v4l2loopback device

On a terminal, run the following.

#### Terminal 0

```bash
export JPG_WIDTH=640
export JPG_HEIGHT=480
gst-launch-1.0 -v v4l2src device=/dev/video0 ! 'video/x-raw, format=YUY2, width=640, height=480, framerate=30/1' ! nvvidconv ! "video/x-raw(memory:NVMM), height=${JPG_HEIGHT}, width=${JPG_WIDTH}, format=I420" ! nvjpegenc ! multipartmux ! multipartdemux single-stream=1 ! "image/jpeg, width=${JPG_WIDTH}, height=${JPG_HEIGHT}, parsed=(boolean)true, colorimetry=(string)2:4:7:1, framerate=(fraction)30/1,sof-marker=(int)0" ! v4l2sink device=/dev/video10
```
This will keep running, so leave the terminal open.


## 

Example of running two containers;

- one for demo-ing `jetnet demo jetnet.yolox.YOLOX_NANO_TRT_FP16` and,
- the other for demo-ing `jetnet.easyocr.EASYOCR_EN_TRT_FP16`

#### Terminal 1

```bash
cd jetnet
sudo docker run \
    --network host \
    --gpus all \
    --runtime nvidia \
    -it \
    --rm \
    --name=jetnet1 \
    -v $(pwd):/jetnet \
    --device /dev/video10 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    jaybdub/jetnet:l4t-35.1.0 \
    /bin/bash -c "cd /jetnet && python3 setup.py develop && jetnet demo --port 8080 --camera_device 10 jetnet.yolox.YOLOX_NANO_TRT_FP16"
```

Open a web browser and access `http://<IP_ADDRESS>:8080`.

> If you are using the same Jetson to run the web browser, it is `http://0.0.0.0:8080`

#### Terminal 2

```bash 
cd jetnet
sudo docker run \
    --network host \
    --gpus all \
    --runtime nvidia \
    -it \
    --rm \
    --name=jetnet2 \
    -v $(pwd):/jetnet \
    --device /dev/video10 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    jaybdub/jetnet:l4t-35.1.0 \
    /bin/bash -c "cd /jetnet && python3 setup.py develop && jetnet demo --port 8081 --camera_device 10 jetnet.easyocr.EASYOCR_EN_TRT_FP16"
```

Open a web browser and access `http://<IP_ADDRESS>:8081`.

> If you are using the same Jetson to run the web browser, it is `http://0.0.0.0:8081`