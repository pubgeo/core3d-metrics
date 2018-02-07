# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = """JHU/APL is supporting the IARPA CORE3D program by providing independent test and evaluation of the performer team solutions for building 3D models based on satellite images and other sources. This is a repository for the metrics being developed to support the program. Performer teams are working with JHU/APL to improve the metrics software and contribute additional metrics that may be used for the program. """

setup(
    name='core3dmetrics',
    version='0.0.0',
    description='JHU/APL Metrics code for IARPA/CORE3D',
    long_description=long_description,
    url='https://github.com/pubgeo/core3d-metric',
    author='JHU/APL',
    author_email='john.doe@jhuapl.edu',
    packages=find_packages(exclude=['aoi-example']),
    include_package_data=True,
    install_requires=['gdal', 'laspy', 'matplotlib', 'numpy', 'scipy'],
    entry_points = {'console_scripts': ['core3d-metrics=core3dmetrics:main']},
    ## entry_points={  # Optional
    ##     'console_scripts': [
    ##         'sample=sample:main',
    ##     ],
    ## },
)
