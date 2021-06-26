# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher

FROM nvidia/cudagl:10.2-devel-ubuntu18.04
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install cudnn
ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 \
    && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3.6-dev python3-pip

RUN pip3 install opencv-python==4.1.0.25 numpy==1.13.3 pandas==0.24.1 networkx==2.2
RUN pip3 install torch==1.6.0 torchvision==0.7.0
# RUN pip3 install transformers==3.1
RUN pip3 install lmdb

# for loading RxR data
RUN pip3 install jsonlines

# Required by Oscar/Transformers codebase
RUN pip3 install boto3 requests regex anytree scikit-image matplotlib pyyaml


RUN pip3 install tensorboardX tqdm wandb

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN ln -sf /usr/bin/python3.6 /usr/bin/python

RUN apt-get update && apt-get install -y tmux expect cmake unzip git
RUN git clone https://github.com/mmurray/cvdn.git /opt/MatterSim
RUN cd /opt/MatterSim && git submodule update --init --recursive && mkdir build && cd build && cmake -DEGL_RENDERING=ON -DPYTHON_INCLUDE_DIR=/usr/include/python -DPYTHON_EXECUTABLE=/usr/bin/python ..
RUN cd /opt/MatterSim/build && make -j8

ENV PYTHONPATH=/opt/MatterSim/build

WORKDIR /root/mount/Matterport3DSimulator
RUN mkdir -p /root/mount/Matterport3DSimulator/tasks/CVDN/data/
RUN mkdir -p /root/mount/Matterport3DSimulator/tasks/NDH/data/

RUN ln -sf /opt/MatterSim/connectivity /root/mount/Matterport3DSimulator/connectivity

# Download and save data in the container
# RUN cd /opt/MatterSim/ && bash tasks/CVDN/data/download.sh && bash tasks/NDH/data/download.sh
# RUN ln -sf /opt/MatterSim/tasks/CVDN/data/*.json /root/mount/Matterport3DSimulator
# RUN ln -sf /opt/MatterSim/tasks/CVDN/data/*.json /root/mount/Matterport3DSimulator/tasks/CVDN/data/
# RUN ln -sf /opt/MatterSim/tasks/NDH/data/*.json /root/mount/Matterport3DSimulator/tasks/NDH/data/
