#
# Align two gridded 3D models using align3d executable.
#

import os
import stat
import numpy as np
import gdal


def align3d(reference_filename, test_filename, exec_path=None):

    # align3d executable (typically on the system $PATH)
    exec_filename = 'align3d'

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
    os.chmod(filename, stat.S_IRGRP | stat.S_IWGRP)
