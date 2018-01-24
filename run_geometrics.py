#
# Run all CORE3D metrics and report results.
#

import os
import sys
import configparser
import gdal
import numpy as np
import geometrics as geo
import argparse



def run_geometrics(configfile):

    # check inputs
    if not os.path.isfile(configfile):
        raise IOError("Configuration file does not exist")

    # parse configuration file
    print("")
    print("Reading configuration from " + configfile)
    print("")
    config = configparser.ConfigParser()
    if len(config.read(configfile)) == 0:
        raise IOError("Unable to read selected .config file")

    # Get test model information from configuration file.
    testDSMFilename = config['INPUT.TEST']['DSMFilename']
    testDTMFilename = config['INPUT.TEST']['DTMFilename']
    testCLSFilename = config['INPUT.TEST']['CLSFilename']
    testMTLFilename = config['INPUT.TEST']['MTLFilename']
    TEST_CLS_VALUE = config.getint('INPUT.TEST', 'CLSMatchValue')

    # Get reference model information from configuration file.
    refDSMFilename = config['INPUT.REF']['DSMFilename']
    refDTMFilename = config['INPUT.REF']['DTMFilename']
    refCLSFilename = config['INPUT.REF']['CLSFilename']
    refNDXFilename = config['INPUT.REF']['NDXFilename']
    refMTLFilename = config['INPUT.REF']['MTLFilename']
    REF_CLS_VALUE = config.getint('INPUT.REF', 'CLSMatchValue')

    # Get material label names and list of material labels to ignore in evaluation.
    materialNames = config['MATERIALS.REF']['MaterialNames'].split(',')
    materialIndicesToIgnore = list(map(int, config['MATERIALS.REF']['MaterialIndicesToIgnore'].split(',')))

    # Register test model to ground truth reference model.
    align3d_path = config['REGEXEPATH']['Align3DPath']
    xyzOffset = geo.align3d(refDSMFilename, testDSMFilename, ExecPath=align3d_path)

    # Read reference model files.
    print("")
    print("Reading reference model files...")
    refMask, tform = geo.imageLoad(refCLSFilename)
    refDSM = geo.imageWarp(refDSMFilename, refCLSFilename)
    refDTM = geo.imageWarp(refDTMFilename, refCLSFilename)

    # Read test model files and apply XYZ offsets.
    print("Reading test model files...")
    print("")
    testDTM = geo.imageWarp(testDTMFilename, refCLSFilename, xyzOffset)
    testMask = geo.imageWarp(testCLSFilename, refCLSFilename, xyzOffset, gdal.gdalconst.GRA_NearestNeighbour)
    testDSM = geo.imageWarp(testDSMFilename, refCLSFilename, xyzOffset)
    testDSM = testDSM + xyzOffset[2]
    testDTM = testDTM + xyzOffset[2]

    # Create mask for ignoring points labeled NoData in reference files.
    refDSM_NoDataValue = geo.getNoDataValue(refDSMFilename)
    refDTM_NoDataValue = geo.getNoDataValue(refDTMFilename)
    refCLS_NoDataValue = geo.getNoDataValue(refCLSFilename)
    ignoreMask = np.zeros_like(refMask, np.bool)

    if refDSM_NoDataValue is not None:
        ignoreMask[refDSM == refDSM_NoDataValue] = True
    if refDTM_NoDataValue is not None:
        ignoreMask[refDTM == refDTM_NoDataValue] = True
    if refCLS_NoDataValue is not None:
        ignoreMask[refMask == refCLS_NoDataValue] = True

    # If quantizing to voxels, then match vertical spacing to horizontal spacing.
    QUANTIZE = config.getboolean('OPTIONS', 'QuantizeHeight')
    if QUANTIZE:
        unitHgt = (np.abs(tform[1]) + abs(tform[5])) / 2
        refDSM = np.round(refDSM / unitHgt) * unitHgt
        refDTM = np.round(refDTM / unitHgt) * unitHgt
        testDSM = np.round(testDSM / unitHgt) * unitHgt
        testDTM = np.round(testDTM / unitHgt) * unitHgt

    # Run the threshold geometry metrics and report results.
    geo.run_threshold_geometry_metrics(refDSM, refDTM, refMask, REF_CLS_VALUE, testDSM, testDTM, testMask, TEST_CLS_VALUE,
                                       tform, xyzOffset, testDSMFilename, ignoreMask)

    # Run the threshold material metrics and report results.
    geo.run_material_metrics(refNDXFilename, refMTLFilename, testMTLFilename, materialNames, materialIndicesToIgnore)


# command line function
def main():

  # parse inputs
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', dest='config', 
      help='Configuration file', required=True)
  
  args = parser.parse_args()
  run_geometrics(configfile=args.config)


if __name__ == "__main__":
  main()

