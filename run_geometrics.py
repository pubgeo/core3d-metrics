#
# Run all CORE3D metrics and report results.
#

import os
import sys
import shutil
import gdalconst
import numpy as np
import geometrics as geo
import argparse
import json


# PRIMARY FUNCTION: RUN_GEOMETRICS
def run_geometrics(configfile,refpath=None,testpath=None,outputpath=None,align=True):

    # check inputs
    if not os.path.isfile(configfile):
        raise IOError("Configuration file does not exist")

    if outputpath is not None and not os.path.isdir(outputpath):
        raise IOError('"outputpath" not a valid folder <{}>'.format(outputpath))

    # parse configuration
    configpath = os.path.dirname(configfile)

    config = geo.parse_config(configfile,
        refpath=(refpath or configpath), 
        testpath=(testpath or configpath))

    # Get test model information from configuration file.
    testDSMFilename = config['INPUT.TEST']['DSMFilename']
    testDTMFilename = config['INPUT.TEST']['DTMFilename']
    testCLSFilename = config['INPUT.TEST']['CLSFilename']
    testMTLFilename = config['INPUT.TEST'].get('MTLFilename',None)

    # Get reference model information from configuration file.
    refDSMFilename = config['INPUT.REF']['DSMFilename']
    refDTMFilename = config['INPUT.REF']['DTMFilename']
    refCLSFilename = config['INPUT.REF']['CLSFilename']
    refNDXFilename = config['INPUT.REF']['NDXFilename']
    refMTLFilename = config['INPUT.REF']['MTLFilename']

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
    if not align:
        print('\nSKIPPING REGISTRATION')
        xyzOffset = (0.0,0.0,0.0)
    else:
        print('\n=====REGISTRATION====='); sys.stdout.flush()
        try:
            align3d_path = config['REGEXEPATH']['Align3DPath']
        except:
            align3d_path = None
        xyzOffset = geo.align3d(refDSMFilename, testDSMFilename_copy, exec_path=align3d_path)

    # Explicitly assign an new no data value to warped images to track filled pixels
    noDataValue = -9999
    
    # Read reference model files.
    print("")
    print("Reading reference model files...")
    refCLS, tform = geo.imageLoad(refCLSFilename)
    refDSM = geo.imageWarp(refDSMFilename, refCLSFilename, noDataValue=noDataValue)
    refDTM = geo.imageWarp(refDTMFilename, refCLSFilename, noDataValue=noDataValue)
    refNDX = geo.imageWarp(refNDXFilename, refCLSFilename, interp_method=gdalconst.GRA_NearestNeighbour).astype(np.uint16)
    refMTL = geo.imageWarp(refMTLFilename, refCLSFilename, interp_method=gdalconst.GRA_NearestNeighbour).astype(np.uint8)

    # Read test model files and apply XYZ offsets.
    print("Reading test model files...")
    print("")
    testDTM = geo.imageWarp(testDTMFilename, refCLSFilename, xyzOffset, noDataValue=noDataValue)
    testCLS = geo.imageWarp(testCLSFilename, refCLSFilename, xyzOffset, gdalconst.GRA_NearestNeighbour)
    testDSM = geo.imageWarp(testDSMFilename, refCLSFilename, xyzOffset, noDataValue=noDataValue)

    if testMTLFilename:
        testMTL = geo.imageWarp(testMTLFilename, refCLSFilename, xyzOffset, gdalconst.GRA_NearestNeighbour).astype(np.uint8)

    testDSM = testDSM + xyzOffset[2]
    testDTM = testDTM + xyzOffset[2]

    # object masks based on CLSMatchValue(s)
    refMask = np.zeros_like(refCLS, np.bool)
    for v in config['INPUT.REF']['CLSMatchValue']:
        refMask[refCLS == v] = True

    testMask = np.zeros_like(testCLS, np.bool)
    for v in config['INPUT.TEST']['CLSMatchValue']:
        testMask[testCLS == v] = True    

    # Create mask for ignoring points labeled NoData in reference files.
    refDSM_NoDataValue = geo.getNoDataValue(refDSMFilename)
    refDTM_NoDataValue = geo.getNoDataValue(refDTMFilename)
    refCLS_NoDataValue = geo.getNoDataValue(refCLSFilename)
    ignoreMask = np.zeros_like(refCLS, np.bool)

    if refDSM_NoDataValue is not None:
        ignoreMask[refDSM == refDSM_NoDataValue] = True
    if refDTM_NoDataValue is not None:
        ignoreMask[refDTM == refDTM_NoDataValue] = True
    if refCLS_NoDataValue is not None:
        ignoreMask[refCLS == refCLS_NoDataValue] = True

    # If quantizing to voxels, then match vertical spacing to horizontal spacing.
    QUANTIZE = config['OPTIONS']['QuantizeHeight']
    if QUANTIZE:
        unitHgt = geo.getUnitHeight(tform)
        refDSM = np.round(refDSM / unitHgt) * unitHgt
        refDTM = np.round(refDTM / unitHgt) * unitHgt
        testDSM = np.round(testDSM / unitHgt) * unitHgt
        testDTM = np.round(testDTM / unitHgt) * unitHgt

        
    if PLOTS_ENABLE:
        
        # geo.imwarp sets no data value to -9999.  Adjust for offset and quantization
        newTestFillValue = noDataValue+xyzOffset[2]
        newRefFillValue  = noDataValue
        if QUANTIZE:
            newTestFillValue = np.round(newTestFillValue / unitHgt) * unitHgt
            newRefFillValue = np.round(newRefFillValue / unitHgt) * unitHgt
    
        plot.make(refMask, 'refMask', 111)
        plot.make(refDSM, 'refDSM', 112, colorbar=True, badValue=newRefFillValue)
        plot.make(refDTM, 'refDTM', 113, colorbar=True, badValue=newRefFillValue)

        plot.make(testMask, 'testMask', 151)
        plot.make(testDSM, 'testDSM', 152, colorbar=True, badValue=newTestFillValue)
        plot.make(testDTM, 'testDSM', 153, colorbar=True, badValue=newTestFillValue)

        plot.make(ignoreMask, 'ignoreMask', 181)


    # Run the threshold geometry metrics and report results.
    metrics = dict()
    metrics['threshold_geometry'] = geo.run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask,
                                       tform, ignoreMask, plot=plot)

    # Run the terrain model metrics and report results.
    try:
        dtm_z_threshold = config['OPTIONS']['TerrainZErrorThreshold']
    except:
        dtm_z_threshold = 1
    metrics['terrain_accuracy'] = geo.run_terrain_accuracy_metrics(refDTM, testDTM, refMask, testMask, dtm_z_threshold, geo.getUnitArea(tform), plot=plot)

    metrics['relative_accuracy'] = geo.run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, plot=plot)

    metrics['offset'] = xyzOffset
    
    fileout = os.path.join(outputpath,os.path.basename(testDSMFilename) + "_metrics.json")
    with open(fileout,'w') as fid:
        json.dump(metrics,fid,indent=2)
    print(json.dumps(metrics,indent=2))

    # Run the threshold material metrics and report results.
    if testMTLFilename:
        geo.run_material_metrics(refNDX, refMTL, testMTL, materialNames, materialIndicesToIgnore)
    else:
        print('WARNING: No test MTL file, skipping material metrics')

    #  If displaying figures, wait for user before existing
    if PLOTS_SHOW:
            input("Press Enter to continue...")

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

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--align', dest='align', action='store_true')
    group.add_argument('--no-align', dest='align', action='store_false')
    group.set_defaults(align=True)

    args = parser.parse_args()

    # gather optional arguments
    kwargs = {}
    if args.refpath: kwargs['refpath'] = args.refpath
    if args.testpath: kwargs['testpath'] = args.testpath
    if args.outputpath: kwargs['outputpath'] = args.outputpath

    # run process
    run_geometrics(configfile=args.config,align=args.align,**kwargs)


if __name__ == "__main__":
    main()

