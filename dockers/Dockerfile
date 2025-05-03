FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libx11-6 libgomp1 python3 python3-pip net-tools iputils-ping iproute2

COPY dll/* /usr/lib/

ENV LD_LIBRARY_PATH=/usr/lib

COPY requirements.txt /

RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]