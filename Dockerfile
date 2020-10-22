FROM jhuapl/pubgeo:latest

RUN apt update && apt upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
	python3 \
	python3-pip \
	python3-gdal \
	python3-tk \
	python3-scipy

RUN apt autoremove -y && rm -rf /var/lib/apt/lists/*
RUN pip3 install "matplotlib==3.0.3" laspy setuptools "jsonschema==2.6.0" "numpy==1.16.2" "opencv-python==4.0.0.21" "Pillow" wheel simplekml tqdm mathutils bpy-cuda bpy_post_install mathutils
WORKDIR /

ARG DOCKER_DEPLOY=true
ENV DOCKER_DEPLOY=$DOCKER_DEPLOY
RUN if [ "$DOCKER_DEPLOY" = true ] ; then \
	pip3 install --no-deps git+https://github.com/Sean-S-Wang/core3d-metrics.git@metrics-dev; \
    fi 

RUN apt purge -y \
    git

ADD entrypoint.bsh /
RUN chmod 755 /entrypoint.bsh
ENTRYPOINT ["/entrypoint.bsh"]

CMD ["echo", "Please run GeoMetrics with a valid AOI configuration", \
    "\ndocker run --rm -v /home/ubuntu/data:/data jhuapl/geometrics core3dmetrics -c <aoi config>"]
