import numpy as np
import os
import json
import math
from scipy.ndimage.measurements import label
from scipy.stats import pearsonr

from .metrics_util import calcMops
from .metrics_util import getUnitArea


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask,
                                   tform, ignoreMask, plot=None, for_objectwise=False, testCONF=None, verbose=True):
    # INPUT PARSING==========

    # parse plot input
    if plot is None or for_objectwise is True:
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

    # refDTM is purposfully used twice for consistency
    test_height = testDSM.astype(np.float64) - refDTM.astype(np.float64)
    # TestDTM is used here to make the images correct
    # test_height_for_image = testDSM.astype(np.float64) - testDTM.astype(np.float64)
    test_height[~test_footprint] = 0
    #test_height_for_image[~test_footprint] = 0

    # total 2D area (in pixels)
    ref_total_area = np.sum(ref_footprint, dtype=np.uint64)
    test_total_area = np.sum(test_footprint, dtype=np.uint64)

    # total 3D volume (in meters^3)
    ref_total_volume = np.sum(np.absolute(ref_height)) * unitArea
    test_total_volume = np.sum(np.absolute(test_height)) * unitArea

    # verbose reporting
    if verbose:
        print('REF height range [mn,mx] = [{},{}]'.format(np.amin(ref_height),np.amax(ref_height)))
        print('TEST height range [mn,mx] = [{},{}]'.format(np.amin(test_height),np.amax(test_height)))
        print('REF area (px), volume (m^3) = [{},{}]'.format(ref_total_area,ref_total_volume))
        print('TEST area (px), volume (m^3) =  [{},{}]'.format(test_total_area,test_total_volume))

    # plot
    error_height_fn = None
    if PLOTS_ENABLE:
        print('Input plots...')

        plot.make(ref_footprint, 'Reference Object Regions', 211, saveName=PLOTS_SAVE_PREFIX+"refObjMask")
        plot.make(ref_height, 'Reference Object Height', 212, saveName=PLOTS_SAVE_PREFIX+"refObjHgt", colorbar=True)

        plot.make(test_footprint, 'Test Object Regions', 251, saveName=PLOTS_SAVE_PREFIX+"testObjMask")
        plot.make(test_height, 'Test Object Height', 252, saveName=PLOTS_SAVE_PREFIX+"testObjHgt", colorbar=True)

        errorMap = (test_height-ref_height)
        errorMap[~ref_footprint & ~test_footprint] = np.nan
        plot.make(errorMap, 'Height Error', 291, saveName=PLOTS_SAVE_PREFIX+"errHgt", colorbar=True)
        plot.make(errorMap, 'Height Error (clipped)', 292, saveName=PLOTS_SAVE_PREFIX+"errHgtClipped", colorbar=True,
                  vmin=-5, vmax=5)
        error_height_fn = plot.make_error_map(error_map=errorMap, ref=ref_footprint, saveName=PLOTS_SAVE_PREFIX+"errHgtImageOnly",
                            ignore=ignoreMask)

    # 2D ANALYSIS==========

    # 2D metric arrays
    tp_2D_array = test_footprint & ref_footprint
    fn_2D_array = ~test_footprint & ref_footprint
    fp_2D_array = test_footprint & ~ref_footprint

    # 2D total area (in pixels)
    tp_total_area = np.sum(tp_2D_array, dtype=np.uint64)
    fn_total_area = np.sum(fn_2D_array, dtype=np.uint64)
    fp_total_area = np.sum(fp_2D_array, dtype=np.uint64)

    # error check (exact, as this is an integer comparison)
    if (tp_total_area + fn_total_area) != ref_total_area:
        raise ValueError('2D TP+FN ({}+{}) does not equal ref area ({})'.format(
            tp_total_area, fn_total_area, ref_total_area))
    elif (tp_total_area + fp_total_area) != test_total_area:
        raise ValueError('2D TP+FP ({}+{}) does not equal test area ({})'.format(
            tp_total_area, fp_total_area, test_total_area))

    # verbose reporting
    if verbose:
        print('2D TP+FN ({}+{}) equals ref area ({})'.format(
            tp_total_area, fn_total_area, ref_total_area))
        print('2D TP+FP ({}+{}) equals test area ({})'.format(
            tp_total_area, fp_total_area, test_total_area))

    # plot
    stoplight_fn = None
    if PLOTS_ENABLE:
        print('2D analysis plots...')
        plot.make(tp_2D_array, 'True Positive Regions',  283, saveName=PLOTS_SAVE_PREFIX+"truePositive")
        plot.make(fn_2D_array, 'False Negative Regions', 281, saveName=PLOTS_SAVE_PREFIX+"falseNegative")
        plot.make(fp_2D_array, 'False Positive Regions', 282, saveName=PLOTS_SAVE_PREFIX+"falsePositive")
        stoplight_fn = plot.make_stoplight_plot(fp_image=fp_2D_array, fn_image=fn_2D_array, ref=ref_footprint, saveName=PLOTS_SAVE_PREFIX+"stoplight")

        layer = np.zeros_like(ref_footprint).astype(np.uint8) + 3  # Initialize as True Negative
        layer[tp_2D_array] = 1  # TP
        layer[fp_2D_array] = 2  # FP
        layer[fn_2D_array] = 4  # FN
        cmap = [[1, 1, 1],  # TP
                [0, 0, 1],  # FP
                [1, 1, 1],  # TN
                [1, 0, 0]]  # FN
        plot.make(layer, 'Object Footprint Errors', 293, saveName=PLOTS_SAVE_PREFIX + "errFootprint", colorbar=False, cmap=cmap)

    # 3D ANALYSIS==========

    # Flip underground reference structures
    # flip all heights where ref_height is less than zero, allowing subsequent calculations
    # to only consider difference relative to positive/absolute reference structures
    tf = ref_height < 0
    ref_height[tf] = -ref_height[tf]
    test_height[tf] = -test_height[tf]

    # separate test height into above & below ground sets
    test_above = np.copy(test_height)
    test_above[test_height < 0] = 0

    test_below = np.copy(test_height)
    test_below[test_height > 0] = 0
    test_below = np.absolute(test_below)

    # 3D metric arrays
    tp_3D_array = np.minimum(ref_height, test_above) # ref/test height overlap
    fn_3D_array = (ref_height - tp_3D_array) # test too short
    fp_3D_array = (test_above - tp_3D_array) + test_below # test too tall OR test below ground

    # 3D metric total volume (in meters^3)
    tp_total_volume = np.sum(tp_3D_array)*unitArea
    fn_total_volume = np.sum(fn_3D_array)*unitArea
    fp_total_volume = np.sum(fp_3D_array)*unitArea

    # error check (floating point comparison via math.isclose)
    if not math.isclose((tp_total_volume + fn_total_volume), ref_total_volume):
        raise ValueError('3D TP+FN ({}+{}) does not equal ref volume ({})'.format(
            tp_total_volume, fn_total_volume, ref_total_volume))
    elif not math.isclose((tp_total_volume + fp_total_volume), test_total_volume):
        raise ValueError('3D TP+FP ({}+{}) does not equal test volume ({})'.format(
            tp_total_volume, fp_total_volume, test_total_volume))

    # verbose reporting
    if verbose:
        print('3D TP+FN ({}+{}) equals ref volume ({})'.format(
            tp_total_volume, fn_total_volume, ref_total_volume))
        print('3D TP+FP ({}+{}) equals test volume ({})'.format(
            tp_total_volume, fp_total_volume, test_total_volume))

    # Confidence Metrics
    if testCONF is not None:
        # check for all common NODATA values
        def nodata_to_nan(img):
            nodata = -9999
            img[img == nodata] = np.nan
            nodata = -10000
            img[img == nodata] = np.nan
            return (img)

        # compute differences
        testDSM_filt = nodata_to_nan(testDSM.copy())
        refDSM_filt = nodata_to_nan(refDSM.copy())
        testCONF_filt = nodata_to_nan(testCONF.copy())
        valid_mask = np.logical_not(np.logical_or(np.logical_or(np.isnan(refDSM_filt), np.isnan(testDSM_filt)), np.isnan(testCONF_filt)))
        building_mask = np.logical_or(test_footprint, ref_footprint)
        w = np.logical_and(valid_mask, building_mask)

        # since not running registration for now, remove z offset
        dz = np.nanmedian(refDSM_filt - testDSM_filt)
        tgt_dsm = testDSM_filt + dz
        dz = np.nanmedian(refDSM_filt - tgt_dsm)
        print('dz after align = ', dz)

        abs_delta = np.abs(testDSM_filt - refDSM_filt)
        abs_delta_minus_mask = abs_delta[w]
        confidence_minus_mask = testCONF_filt[w]
        # compute pearson coefficient
        p = pearsonr(abs_delta_minus_mask, -confidence_minus_mask)
        print(p)
    else:
        p = [np.nan, np.nan]

    # CLEANUP==========

    # final metrics
    metrics = {
        '2D': calcMops(tp_total_area, fn_total_area, fp_total_area),
        '3D': calcMops(tp_total_volume, fn_total_volume, fp_total_volume),
        'area': {'reference_area': np.int(ref_total_area), 'test_area': np.int(test_total_area)},
        'volume': {'reference_volume': np.float(ref_total_volume), 'test_volume': np.float(test_total_volume)},
        'pearson': {'pearson-r': np.float(p[0]), 'pearson-pvalue': np.float(p[1])}
    }

    # verbose reporting
    if verbose:
        print('METRICS REPORT:')
        print(json.dumps(metrics, indent=2))

    # return metric dictionary
    return metrics, unitArea, stoplight_fn, error_height_fn
