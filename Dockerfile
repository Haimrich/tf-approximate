FROM tensorflow/tensorflow:2.1.0-py3

RUN apt-get update
RUN apt-get install -y git cmake

WORKDIR /app/
COPY . .

WORKDIR /app/tf2/build
RUN cmake .. -DTFAPPROX_FORCE_REF_CONV_CPU=ON; make

ENV PYTHONPATH=$PYTHONPATH:/app/tf2/python
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/tf2/build

WORKDIR /app/