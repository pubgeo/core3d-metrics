# JHU/APL pubgeo
JHU/APL is working to help advance the state of the art in geospatial computer vision by developing public benchmark data sets and open source software.
For more information on this and other efforts, please visit [JHU/APL](http://www.jhuapl.edu/pubgeo.html).

## core3d-metrics
 JHU/APL is supporting the IARPA CORE3D program by providing independent test and evaluation of the performer team solutions for building 3D models based on satellite images and other sources. This is a repository for the metrics being developed to support the program. Performer teams are working with JHU/APL to improve the metrics software and contribute additional metrics that may be used for the program.
 
 Preliminary metrics are described in the following paper:
 
 M. Bosch, A. Leichtman, D. Chilcott, H. Goldberg, M. Brown. “Metric Evaluation Pipeline for 3D Modeling of Urban Scenes”, ISPRS Archives, 2017 [pdf](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-1-W1/239/2017/isprs-archives-XLII-1-W1-239-2017.pdf).

### Requirements
The following python3 libraries (and their dependencies) are required:
* gdal
* laspy
* matplotlib
* numpy
* scipy
* tk

Users must also have installed pubgeo's [ALIGN3D](https://github.com/pubgeo/pubgeo/#align3d) software. Software must be compiled and available on $PATH. 

Alternatively, you can use the provided docker [container](Dockerfile).

### core3d-metrics Usage
    python3 run_geometrics.py <AOI Configuration>
One of the first steps is to align your dataset to the ground truth. This is performed using pubgeo's [ALIGN3D](https://github.com/pubgeo/pubgeo/#align3d) algorithm.
The algorithm then calculates metrics for 2D, 3D, and spectral classification against the ground truth.

#### Input
_AOI Configuration_ is a configuration file using python's ConfigParser that is further described in [aoi-config.md].
This configuration file defines which files to analyze and what to compare against (ground truth). Additionally the config is
to toggle various software settings.

#### Example Output
    python3 run_geometrics.py aoi.config
This command would perform metric analysis on the test dataset provided by the aoi.config file. This analysis will also generate the following files (in place):
* < test dataset >_2d_metrics.txt
* < test dataset >_3d_metrics.txt
These files contain the determined metrics for completeness, correctness, f-score, Jaccard Index, Branching Factor, and the Align3d offsets.
