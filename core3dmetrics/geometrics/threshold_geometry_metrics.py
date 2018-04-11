import numpy as np
import os
import json

from .metrics_util import calcMops
from .metrics_util import getUnitArea


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask,
                                   tform, ignoreMask, plot=None, verbose=True):
                    

    # INPUT PARSING==========

    # parse plot input
    if plot is None:
        PLOTS_ENABLE = False
    else:
        PLOTS_ENABLE = True
        PLOTS_SAVE_PREFIX = "thresholdGeometry_"    
                                   
    # Determine evaluation units.
    unitArea = getUnitArea(tform)
                                 
    # 2D footprints for evaluation
    ref_footprint = refMask & ~ignoreMask
    test_footprint = testMask & ~ignoreMask

    # building height (DSM-DTM, with zero elevation outside footprint)
    ref_height = refDSM.astype(np.float64) - refDTM.astype(np.float64)
    ref_height[~ref_footprint] = 0

    test_height = testDSM.astype(np.float64) - testDTM.astype(np.float64)
    test_height[~test_footprint] = 0

    # total 2D area
    REF_2D = np.sum(ref_footprint)
    TEST_2D = np.sum(test_footprint)

    # total 3D volume 
    REF_3D = np.sum(np.absolute(ref_height)) * unitArea
    TEST_3D = np.sum(np.absolute(test_height)) * unitArea

    # verbose reporting
    if verbose:
        print('REF height range [mn,mx] = [{},{}]'.format(np.amin(ref_height),np.amax(ref_height)))
        print('TEST height range [mn,mx] = [{},{}]'.format(np.amin(test_height),np.amax(test_height)))
        print('REF area (px), volume (m^3) = [{},{}]'.format(REF_2D,REF_3D))
        print('TEST area (px), volume (m^3) =  [{},{}]'.format(TEST_2D,TEST_3D))

    # plot 
    if PLOTS_ENABLE:
        print('Input plots...')

        plot.make(ref_footprint, 'Reference Object Regions', 211, saveName=PLOTS_SAVE_PREFIX+"refObjMask")
        plot.make(ref_height, 'Reference Object Height', 212, saveName=PLOTS_SAVE_PREFIX+"refObjHgt", colorbar=True)
        
        plot.make(test_footprint, 'Test Object Regions', 251, saveName=PLOTS_SAVE_PREFIX+"testObjHgt")
        plot.make(test_height, 'Test Object Height', 252, saveName=PLOTS_SAVE_PREFIX+"testObjHgt", colorbar=True)

        errorMap = (test_height-ref_height)
        errorMap[~ref_footprint & ~test_footprint] = np.nan
        plot.make(errorMap, 'Height Error', 291, saveName=PLOTS_SAVE_PREFIX+"errHgt", colorbar=True)
        plot.make(errorMap, 'Height Error (clipped)', 292, saveName=PLOTS_SAVE_PREFIX+"errHgtClipped", colorbar=True,
            vmin=-5,vmax=5)


    # 2D ANALYSIS==========

    # 2D metrics
    tp_2D =  test_footprint &  ref_footprint
    fn_2D = ~test_footprint &  ref_footprint
    fp_2D =  test_footprint & ~ref_footprint
    
    TP_2D = np.sum(tp_2D)
    FN_2D = np.sum(fn_2D)
    FP_2D = np.sum(fp_2D)

    # error check:
    if (TP_2D + FN_2D) != REF_2D:
        raise ValueError('2D TP+FN ({}+{}) does not equal ref area ({})'.format(
            TP_2D, FN_2D, REF_2D))
    elif (TP_2D + FP_2D) != TEST_2D:
        raise ValueError('2D TP+FP ({}+{}) does not equal test area ({})'.format(
            TP_2D, FP_2D, TEST_2D))
    
    # verbose reporting
    if verbose:
        print('2D TP+FN ({}+{}) equals ref area ({})'.format(
            TP_2D, FN_2D, REF_2D))
        print('2D TP+FP ({}+{}) equals test area ({})'.format(
            TP_2D, FP_2D, TEST_2D))

    # plot
    if PLOTS_ENABLE:   
        print('2D analysis plots...')
        plot.make(tp_2D, 'True Positive Regions',  283, saveName=PLOTS_SAVE_PREFIX+"truePositive")
        plot.make(fn_2D, 'False Negative Regions', 281, saveName=PLOTS_SAVE_PREFIX+"falseNegetive")
        plot.make(fp_2D, 'False Positive Regions', 282, saveName=PLOTS_SAVE_PREFIX+"falsePositive")


    # 3D ANALYSIS==========

    # Flip underground reference structures
    # flip all heights where ref_height is less than zero, allowing subsequent calculations
    # to only consider difference relative to positive/absolute reference structures
    tf = ref_height < 0
    ref_height[tf] = -ref_height[tf]
    test_height[tf] = -test_height[tf]

    # separate test height into above & below ground sets
    test_above = np.copy(test_height)
    test_above[test_height<0] = 0

    test_below = np.copy(test_height)
    test_below[test_height>0] = 0
    test_below = np.absolute(test_below)

    # 3D metrics
    tp_3D = np.minimum(ref_height,test_above) # ref/test height overlap
    fn_3D = (ref_height - tp_3D) # test too short
    fp_3D = (test_above - tp_3D) + test_below # test too tall OR test below ground
    
    TP_3D = np.sum(tp_3D)*unitArea
    FN_3D = np.sum(fn_3D)*unitArea
    FP_3D = np.sum(fp_3D)*unitArea

    # error check:
    if (TP_3D + FN_3D) != REF_3D:
        raise ValueError('3D TP+FN ({}+{}) does not equal ref volume ({})'.format(
            TP_3D, FN_3D, REF_3D))
    elif (TP_3D + FP_3D) != TEST_3D:
        raise ValueError('3D TP+FP ({}+{}) does not equal test volume ({})'.format(
            TP_3D, FP_3D, TEST_3D))

    # verbose reporting
    if verbose:
        print('3D TP+FN ({}+{}) equals ref volume ({})'.format(
            TP_3D, FN_3D, REF_3D))
        print('3D TP+FP ({}+{}) equals test volume ({})'.format(
            TP_3D, FP_3D, TEST_3D))


    # CLEANUP==========

    # final metrics
    metrics = {
        '2D': calcMops(TP_2D, FN_2D, FP_2D),
        '3D': calcMops(TP_3D, FN_3D, FP_3D),
    }

    # verbose reporting
    if verbose:
        print('METRICS REPORT:')
        print(json.dumps(metrics,indent=2))

    # return metric dictionary
    return metrics
