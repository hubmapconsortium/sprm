FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install \
    python3 \
    python3-pip\
 && rm -rf /var/cache/apt/*

COPY docker/spatialdata-conversion/requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

WORKDIR /opt

COPY sprm/SPRM_output_convert.py /opt

ENV OPENBLAS_NUM_THREADS=64

CMD ["/bin/bash"]
