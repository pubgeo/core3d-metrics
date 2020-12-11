#
# Align two gridded 3D models using align3d executable.
#

import os
import platform
import numpy as np
import gdal
from utils.align3d import AlignParameters, AlignTarget2Reference
from datetime import datetime
import sys
from pathlib import Path


def align3d_python(reference_filename, target_filename, gsd=1.0, maxt=10.0, maxdz=0.0):
    """
    Runs the python port of align3d
    :param reference_filename: ground truth reference file
    :param target_filename: target/performer file
    :param gsd: Ground Sample Distance (GSD) for gridding point cloud (meters); default = 1.0
    :param maxt: Maximum horizontal translation in search (meters); default = 10.0
    :param maxdz: Max local Z difference (meters) for matching; default = 2*gsd
    :return: xyz offsets
    """

    offset= None
    params = AlignParameters()
    params.gsd = gsd
    params.maxt = maxt
    params.maxdz = maxdz
    # Default MAXDZ = GSD x 2 to ensure reliable performance on steep slopes
    if params.maxdz == 0.0:
        params.maxdz = params.gsd * 2.0
    print("Selected Parameters:")
    print(f" ref   = {reference_filename}")
    print(f" tgt   = {target_filename}")
    print(f" gsd   = {params.gsd}")
    print(f" maxdz = {params.maxdz}")
    print(f" maxt  = {params.maxt}")

    # Intialiize timer
    start_time = datetime.now()
    try:
        AlignTarget2Reference(Path(reference_filename), Path(target_filename), params)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    end_time = datetime.now() - start_time
    print(f" Total time elapsed = {end_time} seconds")

    registered_filename = os.path.join(target_filename[0:-4] + '_aligned.tif')
    offset_filename = os.path.join(target_filename[0:-4] + '_offsets.txt')

    # Open permissions on output files
    unroot(registered_filename)
    unroot(offset_filename)

    # Read XYZ offset from align3d output file.
    offsets = readXYZoffset(offset_filename)

    return offset

def align3d(reference_filename, test_filename, exec_path=None):

    # align3d executable (typically on the system $PATH)
    exec_filename = 'align3d'
    if platform.system() == "Windows":
        exec_filename = exec_filename + ".exe"

    # locate align3d executable
    if exec_path: 
        exec_filename = os.path.abspath(os.path.join(exec_path,exec_filename))
        if not os.path.isfile(exec_filename):
            raise IOError('"align3d" executable not found at <{}>'.format(exec_filename))

    # In case file names have relative paths, convert to absolute paths.
    reference_filename = os.path.abspath(reference_filename)
    test_filename = os.path.abspath(test_filename)

    # Run align3d.
    command = exec_filename + " " + reference_filename + " " + test_filename + ' maxt=10.0'
    print("")
    print("Registering test model to reference model to determine XYZ offset.")
    print("")
    print(command)
    print("")
    os.system(command)

    # Names of files produced by registration process
    registered_filename = os.path.join(test_filename[0:-4] + '_aligned.tif')
    offset_filename = os.path.join(test_filename[0:-4] + '_offsets.txt')

    # TODO: This is here for docker scenarios where new files are owned by root
    # Open permissions on output files
    unroot(registered_filename)
    unroot(offset_filename)
    # TODO: registration makes more files that may need 'un-rooting'

    # Read XYZ offset from align3d output file.
    offsets = readXYZoffset(offset_filename)

    return offsets


def readXYZoffset(filename):
    with open(filename, "r") as fid:
        offsetstr = fid.readlines()        
    cc = offsetstr[1].split(' ')
    xyzoffset = [float(v) for v in [cc[0],cc[2],cc[4]]]
    return xyzoffset


def getXYZoffsetFilename(testFilename):
    offset_filename = os.path.join(testFilename[0:-4] + '_offsets.txt')
    return offset_filename


def unroot(filename):
    os.chmod(filename, 0o644)
