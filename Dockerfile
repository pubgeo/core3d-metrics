FROM jhuapl/pubgeo:latest
RUN apt update && apt upgrade -y && apt install -y --fix-missing --no-install-recommends \
    git \
	python3 \
	python3-pip \
	python3-gdal \
	python3-tk \
	python3-scipy
RUN apt autoremove -y && rm -rf /var/lib/apt/lists/*
RUN pip3 install matplotlib laspy setuptools "jsonschema==2.6.0" "numpy>=1.13"
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

CMD echo "Please run GeoMetrics with an AOI configuration"\
    echo "docker run --rm -v /home/ubuntu/annoteGeoExamples:/data jhuapl/geometrics python3 -m core3dmetrics -c <aoi config>"
