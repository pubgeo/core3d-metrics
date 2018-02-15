#
# Run all CORE3D metrics and report results.
#

import os
import sys
import shutil
import configparser
import gdal, gdalconst
import numpy as np
import geometrics as geo
import argparse
import json
import glob


# HELPER: LOCATE ABSOLUTE FILE PATH with GLOB
def findfiles(data,path=None):

    for key,file in data.items():
        if not key.lower().endswith('filename'): continue

        print('\nSearching for "{}"'.format(key))

        # absolute path to file
        if not os.path.isabs(file):
            if path: file = os.path.join(path,file)
            file = os.path.abspath(file)

        # locate file (use glob to allow wildcards)
        files = glob.glob(file)

        if not files:
            print("WARNING: unable to locate file <{}>".format(file))
            file = None
        else:
            if len(files) > 1:
                print('WARNING: multiple files located for <{}>, using 1st file'.format(file))

            file = files[0]
            print('File located <{}>'.format(file))

        # save file to data
        data[key] = file

    return data


# PRIMARY FUNCTION: RUN_GEOMETRICS
def run_geometrics(configfile,refpath=None,testpath=None,outputpath=None):

    # current absolute path of this function
    curpath = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    # check inputs
    if not os.path.isfile(configfile):
        raise IOError("Configuration file does not exist")

    if refpath is None:
        refpath = curpath
    elif not os.path.isdir(refpath):
        raise IOError('"refpath" not a valid folder <{}>'.format(refpath))

    if testpath is None:
        testpath = curpath
    elif not os.path.isdir(refpath):
        raise IOError('"testpath" not a valid folder <{}>'.format(testpath))

    if outputpath is not None and not os.path.isdir(outputpath):
        raise IOError('"outputpath" not a valid folder <{}>'.format(outputpath))


    # parse configuration file
    print("\nReading configuration from <{}>".format(configfile))

    # JSON parsing
    if configfile.endswith(('.json','.JSON')):

        # open & read JSON file
        with open(configfile,'r') as fid:
            config = json.load(fid)

    # CONFIG parsing
    elif configfile.endswith(('.config','.CONFIG')):

        # setup config parser
        parser = configparser.ConfigParser()
        parser.optionxform = str # maintain case-sensitive items

        # read entire configuration file into dict
        if len(parser.read(configfile)) == 0:
            raise IOError("Unable to read selected .config file")
        config = {s:dict(parser.items(s)) for s in parser.sections()}   

        # special section/item parsing
        s = 'INPUT.TEST'; i = 'CLSMatchValue'; config[s][i] = int(config[s][i])
        s = 'INPUT.REF'; i = 'CLSMatchValue'; config[s][i] = int(config[s][i])
        # bool(config[s][i]) does not interpret 'true'/'false' strings
        s = 'OPTIONS'; i = 'QuantizeHeight'; config[s][i] = parser.getboolean(s,i)  
        s = 'PLOTS'; i = 'ShowPlots'; config[s][i] = parser.getboolean(s,i) 
        s = 'PLOTS'; i = 'SavePlots'; config[s][i] = parser.getboolean(s,i)
        s = 'MATERIALS.REF'; i = 'MaterialNames'; config[s][i] = config[s][i].split(',')
        s = 'MATERIALS.REF'; i = 'MaterialIndicesToIgnore'; config[s][i] = list(map(int, config[s][i].split(',')))

    # unrecognized config file type
    else:
        raise IOError('Unrecognized configuration file')


    # locate files for each "xxxFilename" configuration parameter
    # this makes use of "refpath" and "testpath" arguments for relative filenames
    for item in [('INPUT.REF',refpath),('INPUT.TEST',testpath)]:
        sec = item[0]; path = item[1]
        print('\n=====PROCESSING FILES FOR "{}"====='.format(sec))
        config[sec] = findfiles(config[sec],path)


    # print final configuration
    print('\n=====FINAL CONFIGURATION=====')
    print(json.dumps(config,indent=2))


    # Get test model information from configuration file.
    testDSMFilename = config['INPUT.TEST']['DSMFilename']
    testDTMFilename = config['INPUT.TEST']['DTMFilename']
    testCLSFilename = config['INPUT.TEST']['CLSFilename']
    testMTLFilename = config['INPUT.TEST']['MTLFilename']
    TEST_CLS_VALUE = config['INPUT.TEST']['CLSMatchValue']

    # Get reference model information from configuration file.
    refDSMFilename = config['INPUT.REF']['DSMFilename']
    refDTMFilename = config['INPUT.REF']['DTMFilename']
    refCLSFilename = config['INPUT.REF']['CLSFilename']
    refNDXFilename = config['INPUT.REF']['NDXFilename']
    refMTLFilename = config['INPUT.REF']['MTLFilename']
    REF_CLS_VALUE = config['INPUT.REF']['CLSMatchValue']

    # Get material label names and list of material labels to ignore in evaluation.
    materialNames = config['MATERIALS.REF']['MaterialNames']
    materialIndicesToIgnore = config['MATERIALS.REF']['MaterialIndicesToIgnore']
    
    # Get plot settings from configuration file
    PLOTS_SHOW   = config['PLOTS']['ShowPlots']
    PLOTS_SAVE   = config['PLOTS']['SavePlots']
    PLOTS_ENABLE = PLOTS_SHOW or PLOTS_SAVE

    # default output path
    if outputpath is None:
        outputpath = os.path.dirname(testDSMFilename)

    # Configure plotting
    basename = os.path.basename(testDSMFilename)
    if PLOTS_ENABLE:
        plot = geo.plot(saveDir=outputpath, autoSave=PLOTS_SAVE, savePrefix=basename+'_', badColor='black',showPlots=PLOTS_SHOW)
    else:
        plot = None
        
    # copy testDSM to the output path
    # this is a workaround for the "align3d" function with currently always
    # saves new files to the same path as the testDSM
    src = testDSMFilename
    dst = os.path.join(outputpath,os.path.basename(src))
    if not os.path.isfile(dst): shutil.copyfile(src,dst)
    testDSMFilename_copy = dst

    # Register test model to ground truth reference model.
    print('\n=====REGISTRATION====='); sys.stdout.flush()
    align3d_path = config['REGEXEPATH']['Align3DPath']
    xyzOffset = geo.align3d(refDSMFilename, testDSMFilename_copy, exec_path=align3d_path)

    # Explicitly assign an new no data value to warped images to track filled pixels
    noDataValue = -9999
    
    # Read reference model files.
    print("")
    print("Reading reference model files...")
    refMask, tform = geo.imageLoad(refCLSFilename)
    refDSM = geo.imageWarp(refDSMFilename, refCLSFilename, noDataValue=noDataValue)
    refDTM = geo.imageWarp(refDTMFilename, refCLSFilename, noDataValue=noDataValue)
    refNDX = geo.imageWarp(refNDXFilename, refCLSFilename, interp_method=gdalconst.GRA_NearestNeighbour).astype(np.uint16)
    refMTL = geo.imageWarp(refMTLFilename, refCLSFilename, interp_method=gdalconst.GRA_NearestNeighbour).astype(np.uint8)

    # Read test model files and apply XYZ offsets.
    print("Reading test model files...")
    print("")
    testDTM = geo.imageWarp(testDTMFilename, refCLSFilename, xyzOffset, noDataValue=noDataValue)
    testMask = geo.imageWarp(testCLSFilename, refCLSFilename, xyzOffset, gdalconst.GRA_NearestNeighbour)
    testDSM = geo.imageWarp(testDSMFilename, refCLSFilename, xyzOffset, noDataValue=noDataValue)
    testMTL = geo.imageWarp(testMTLFilename, refCLSFilename, xyzOffset, gdalconst.GRA_NearestNeighbour).astype(np.uint8)

    # Apply registration offset, only to valid data to allow better tracking of bad data
    testValidData = (testDSM != noDataValue) & (testDSM != noDataValue)
    testDSM[testValidData] = testDSM[testValidData] + xyzOffset[2]
    testDTM[testValidData] = testDTM[testValidData] + xyzOffset[2]

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
    QUANTIZE = config['OPTIONS']['QuantizeHeight']
    if QUANTIZE:
        unitHgt = (np.abs(tform[1]) + abs(tform[5])) / 2
        refDSM = np.round(refDSM / unitHgt) * unitHgt
        refDTM = np.round(refDTM / unitHgt) * unitHgt
        testDSM = np.round(testDSM / unitHgt) * unitHgt
        testDTM = np.round(testDTM / unitHgt) * unitHgt
        noDataValue = np.round(noDataValue / unitHgt) * unitHgt
        
    if PLOTS_ENABLE:
        # Reference models can bad voids, so ignore bad data on display
        plot.make(refDSM, 'refDSM', 111, colorbar=True, badValue=noDataValue)
        plot.make(refDTM, 'refDTM', 112, colorbar=True, badValue=noDataValue)
        plot.make(refMask, 'refMask', 113)

        # Test models shouldn't have any bad data,
        # so display the bad values to highlight them,
        # unlike with the refSDM/refDTM
        plot.make(testDSM, 'testDSM', 151, colorbar=True)
        plot.make(testDTM, 'testDSM', 152, colorbar=True)
        plot.make(testMask, 'testMask', 153, colorbar=True)

        plot.make(ignoreMask, 'ignoreMask', 181)


    # Run the threshold geometry metrics and report results.
    metrics = geo.run_threshold_geometry_metrics(refDSM, refDTM, refMask, REF_CLS_VALUE, testDSM, testDTM, testMask, TEST_CLS_VALUE,
                                       tform, ignoreMask, plot=plot)

    metrics['registration_offset'] = xyzOffset
    
    fileout = os.path.join(outputpath,os.path.basename(testDSMFilename) + "_metrics.json")
    with open(fileout,'w') as fid:
        json.dump(metrics,fid,indent=2)
    print(json.dumps(metrics,indent=2))

    # Run the threshold material metrics and report results.
    geo.run_material_metrics(refNDX, refMTL, testMTL, materialNames, materialIndicesToIgnore)


# command line function
def main():

  # parse inputs
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', dest='config', 
      help='Configuration file', required=True)
  parser.add_argument('-r', '--reference', dest='refpath', 
      help='Reference data folder', required=False)
  parser.add_argument('-t', '--test', dest='testpath', 
      help='Test data folder', required=False)
  parser.add_argument('-o', '--output', dest='outputpath', 
      help='Output folder', required=False)
  
  args = parser.parse_args()

  # gather optional arguments
  kwargs = {}
  if args.refpath: kwargs['refpath'] = args.refpath
  if args.testpath: kwargs['testpath'] = args.testpath
  if args.outputpath: kwargs['outputpath'] = args.outputpath

  # run process
  run_geometrics(configfile=args.config,**kwargs)


if __name__ == "__main__":
  main()

