#
# Align two gridded 3D models using align3d executable.
#

import os
import stat
import numpy as np
import gdal


def align3d(reference_filename, test_filename, **keyword_parameters):
    # Determine location of the align3d executable.
    if 'ExecPath' in keyword_parameters:
        print("Working Directory parameter specified: ", keyword_parameters['ExecPath'])
        exec_path = keyword_parameters["ExecPath"]
    else:
        print("No working directory parameter specified, default is the current working directory")
        exec_path = os.path.dirname(os.path.realpath(__file__))
    exec_path = os.path.abspath(exec_path)
    exec_filename = os.path.join(exec_path, 'align3d')

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
    xyz_offset = np.zeros([3, 1])
    file_obj = open(filename, "r")
    offset_string = file_obj.readlines()
    cc = offset_string[1].split(' ')
    xyz_offset[0] = cc[0]
    xyz_offset[1] = cc[2]
    xyz_offset[2] = cc[4]
    file_obj.close()
    return xyz_offset


def getXYZoffsetFilename(testFilename):
    offset_filename = os.path.join(testFilename[0:-4] + '_offsets.txt')
    return offset_filename


def unroot(filename):
    os.chmod(filename, stat.S_IRGRP | stat.S_IWGRP)
