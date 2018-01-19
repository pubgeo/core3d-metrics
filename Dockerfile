FROM jhuapl/pubgeo:latest
RUN apt update && apt upgrade -y && apt install -y --fix-missing --no-install-recommends \
    git \
	python3 \
	python3-pip \
	python3-gdal \
	python3-tk \
	python3-scipy
RUN apt autoremove -y && rm -rf /var/lib/apt/lists/*
RUN pip3 install matplotlib laspy "numpy>=1.13"
WORKDIR /
RUN git clone https://github.com/pubgeo/GeoMetrics
RUN apt purge -y \
    git
WORKDIR /GeoMetrics
CMD echo "Please run GeoMetrics with an AOI configuration"\
    echo "docker run --rm -v /home/ubuntu/annoteGeoExamples:/data jhuapl/geometrics python3 run_geometrics.py<aoi config>"
