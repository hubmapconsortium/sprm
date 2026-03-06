FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install \
    python3 \
    python3-pip\
 && rm -rf /var/cache/apt/*

COPY sprm/SPRM_output_convert.py /opt

COPY docker/spatialdata-conversion/requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

WORKDIR /opt
COPY . /opt/sprm
RUN cd sprm \
 && cp sprm-git-revision.json /opt \
 && python3 -m pip install . \
 && cd .. \
 && rm -rf sprm /root/cache/.pip

ENV OPENBLAS_NUM_THREADS=64

CMD ["/bin/bash"]
