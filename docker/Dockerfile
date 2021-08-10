FROM continuumio/anaconda3:2021.05

WORKDIR /opt
COPY . /opt/sprm
RUN cd sprm \
 && python3 -m pip install . \
 && cd .. \
 && rm -rf sprm /root/cache/.pip

CMD ["/bin/bash"]
