FROM nvcr.io/nvidia/pytorch:21.05-py3


RUN pip3 install termcolor

RUN pip3 install git+https://github.com/JaidedAI/EasyOCR.git@v1.5.0
RUN pip3 install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@master
RUN pip3 install pre-commit
RUN pip3 install pydantic
RUN apt update && apt install unzip
RUN pip3 install python-multipart
RUN pip3 install --upgrade typing_extensions
RUN pip3 install fastapi uvicorn[standard]
RUN pip3 install pytest
RUN pip3 install fastapi_socketio
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && python3 setup.py install
RUN pip3 install git+https://github.com/NVIDIA-AI-IOT/trt_pose.git

# MMLAB
RUN apt-get purge cmake -y
ENV CMAKE_VER 3.23.1
ENV ARCH x86_64
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.sh
RUN chmod +x cmake-${CMAKE_VER}-linux-${ARCH}.sh
RUN ./cmake-${CMAKE_VER}-linux-${ARCH}.sh --prefix=/usr --skip-license

# MMLAB/MMCV
RUN apt-get install -y libssl-dev
RUN git clone --branch v1.6.1 https://github.com/open-mmlab/mmcv.git && cd mmcv && MMCV_WITH_OPS=1 FORCE_CUDA=1 python3 setup.py develop
RUN pip install onnx

RUN apt-get update
RUN apt-get install -y libhdf5-dev libhdf5-103 pkg-config 
RUN pip install versioned-hdf5

# MMLAB/PPL.CV
RUN apt-get install -y libspdlog-dev
ENV MMLAB_DIR /mmlab
RUN mkdir -p ${MMLAB_DIR}
RUN cd ${MMLAB_DIR} && git clone https://github.com/openppl-public/ppl.cv.git
ENV PPLCV_DIR /mmlab/ppl.cv
RUN cd ${PPLCV_DIR} && ./build.sh cuda

# MMLAB/MMDEPLOY
ENV MMDEPLOY_DIR ${MMLAB_DIR}/mmdeploy
ENV TENSORRT_DIR /usr/include/x86_64-linux-gnu
RUN mkdir -p ${MMDEPLOY_DIR}
RUN cd ${MMLAB_DIR} && git clone --branch v0.7.0 --recursive https://github.com/open-mmlab/mmdeploy.git
RUN cd ${MMDEPLOY_DIR} && mkdir -p build && cd build && cmake .. -DMMDEPLOY_TARGET_BACKENDS="trt" && make -j$(nproc) && make install
RUN cd ${MMDEPLOY_DIR} && pip install -v -e .


RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH && cd ${MMDEPLOY_DIR} && mkdir -p build && cd build && cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="trt" \
    -DMMDEPLOY_CODEBASES=all \
    -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl && make -j$(nproc) && make install

RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV PYTHONPATH ${MMDEPLOY_DIR}/build/lib

# MMLAB/MMDETECTION
ENV MMDET_DIR ${MMLAB_DIR}/mmdetection
RUN cd ${MMLAB_DIR} && git clone https://github.com/open-mmlab/mmdetection.git
RUN cd $MMDET_DIR && pip install -r requirements/build.txt && pip install -v -e .

## MMLAB/MMOCR
ENV MMOCR_DIR ${MMLAB_DIR}/mmocr
RUN cd ${MMLAB_DIR} && git clone https://github.com/open-mmlab/mmocr.git
RUN cd $MMOCR_DIR && pip install -r requirements/build.txt && pip install -v -e .

RUN pip install progressbar

RUN pip install mkdocs mkdocs-material mkdocs-macros-plugin
RUN pip install pybind11 msgpack