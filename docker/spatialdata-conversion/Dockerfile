FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
 && rm -rf /var/cache/apt/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

COPY docker/spatialdata-conversion/requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

WORKDIR /opt

COPY sprm/SPRM_output_convert.py /opt

ENV OPENBLAS_NUM_THREADS=64

CMD ["/bin/bash"]
