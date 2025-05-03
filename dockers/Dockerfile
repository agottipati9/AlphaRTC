FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libx11-6 libgomp1 python3 net-tools iputils-ping iproute2

# COPY /usr/lib /usr/lib/

# COPY /usr/bin /usr/bin/

# COPY install-build-deps.py /
# COPY install-build-deps.sh /

# RUN chmod +x /install-build-deps.sh
# RUN chmod +x /install-build-deps.py

# RUN /install-build-deps.sh

COPY dll/* /usr/lib/

ENV LD_LIBRARY_PATH=/usr/lib

CMD ["/bin/bash"]