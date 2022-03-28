## core3d-metrics
JHU/APL supported the IARPA CORE3D program by providing independent test and evaluation of the performer team solutions for building 3D models based on satellite images and other sources. Metric evaluation code was maintained here for transparency and to enable collaboration for improvements with performer teams. Legacy MATLAB code is also now archived here for reference. None of this code is being actively maintained at this time.
 
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

Alternatively, you can use the provided docker [container](Dockerfile).

### Installation
Recommend: use a [virtual environment](https://docs.python.org/3/tutorial/venv.html)

    python3 setup.py install
    python3 setup.py install --prefix=$MY_ROOT

### Usage
If installed

    # from command line
    core3d-metrics --help
    core3d-metrics -c <AOI Configuration>
    python3 -m core3dmetrics -c <AOI Configuration>

    # in use code:
    import core3dmetrics.geometrics as geo
    geo.registration.align3d(reference_filename, test_filename)
    core3dmetrics.main(['--help"])

If not installed

    cd core3dmetrics
    python3 run_geometrics.py -c <AOI Configuration> [-o <Output folder>  -r <Reference data folder> -t <Test data folder>]

One of the first steps is to align your dataset to the ground truth. This is performed using pubgeo's [ALIGN3D](https://github.com/pubgeo/pubgeo/#align3d) algorithm.
The algorithm then calculates metrics for 2D, 3D, and spectral classification against the ground truth.

###### Usage Statement
        usage: core3dmetrics [-h] -c  [-r] [-t] [-o] [--align | --no-align] [--test-ignore] [--save-plots | --skip-save-plots] [--save-aligned]
        core3dmetrics entry point
        optional arguments:
          -h, --help         show this help message and exit
          -c , --config      Configuration file
          -r , --reference   Reference data folder
          -t , --test        Test data folder
          -o , --output      Output folder
          --align            Enable alignment (default)
          --no-align         Disable alignment
          --save-aligned     Save aligned images as geoTIFF (not enabled by default)
          --test-ignore      Enable NoDataValue pixels in test CLS image to be 
                             ignored during evaluation
          --save-plots       Enable saving plots (overrides config file setting)
          --skip-save-plots  Disable saving plots (overrides config file setting)                             

#### Input
_AOI Configuration_ is a configuration file using python's ConfigParser that is further described in [aoi-config.md](aoi-example/aoi-config.md).
This configuration file defines which files to analyze and what to compare against (ground truth). Additionally the config is used to toggle various software settings.

#### Example Output
    python3 -m core3dmetrics -c aoi.config
This command would perform metric analysis on the test dataset provided by the aoi.config file. This analysis will also generate the following files (in place):
* < test dataset >_metrics.json

These files contain the determined metrics for completeness, correctness, f-score, Jaccard Index, Branching Factor, and the Align3d offsets.
