FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get install -y git cmake gdb python3-dev python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip install tensorflow-gpu==2.2.0
RUN pip install sklearn pillow opencv-python opencv-python-headless

# COPY .. /app/tf-approximate/
# WORKDIR /app/tf-approximate/tf2/build

## Set the appropriate DTFAPPROX_CUDA_ARCHS for your GPU. See compute capability https://en.wikipedia.org/wiki/CUDA#GPUs_supported
## Ex. GTX 1080 Ti -> 6.1 -> -DTFAPPROX_CUDA_ARCHS="61"

# RUN cmake .. -DTFAPPROX_CUDA_ARCHS="XX" -DTFAPPROX_ALLOW_CPU_CONV=OFF; make -j6

WORKDIR /app/tf-approximate

ENV PYTHONPATH=$PYTHONPATH:/app/tf-approximate/tf2/python
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/tf-approximate/tf2/build