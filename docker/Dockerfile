FROM continuumio/anaconda3:2024.02-1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install libgl1-mesa-glx \
 && rm -rf /var/cache/apt/*

COPY sprm/SPRM_output_convert.py /opt

WORKDIR /opt
COPY . /opt/sprm
RUN cd sprm \
 && cp sprm-git-revision.json /opt \
 && python3 -m pip install . \
 && cd .. \
 && rm -rf sprm /root/cache/.pip

ENV OPENBLAS_NUM_THREADS=64

CMD ["/bin/bash"]
