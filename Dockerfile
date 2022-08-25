FROM jhuapl/pubgeo:latest

RUN apt update && apt upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
	python3 \
    python3-dev \
	python3-pip \
	python3-gdal \
	python3-tk \
	python3-scipy

RUN apt autoremove -y && rm -rf /var/lib/apt/lists/*
RUN pip3 install future-fstrings
RUN pip3 install wheel
RUN pip3 install "matplotlib==3.0.3" laspy setuptools "jsonschema==2.6.0" "numpy==1.16.2" "opencv-python==4.0.0.21" "Pillow" simplekml tqdm "mathutils==2.81.2"
WORKDIR /

ARG DOCKER_DEPLOY=true
ENV DOCKER_DEPLOY=$DOCKER_DEPLOY
RUN if [ "$DOCKER_DEPLOY" = true ] ; then \
	pip3 install --no-deps git+https://github.com/pubgeo/core3d-metrics; \
    fi

RUN apt purge -y \
    git

ADD entrypoint.bsh /
RUN chmod 755 /entrypoint.bsh
ENTRYPOINT ["/entrypoint.bsh"]

CMD ["echo", "Please run GeoMetrics with a valid AOI configuration", \
    "\ndocker run --rm -v /home/ubuntu/data:/data jhuapl/geometrics core3dmetrics -c <aoi config>"]
