#!/bin/bash


docker run \
    --network host \
    --gpus all \
    -it \
    -d \
    --rm \
    --name=jetnet \
    -v $(pwd):/jetnet \
    --device /dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    jaybdub/jetnet:x86-21.05
