import numpy as np
import os

from .metrics_util import calcMops
from .metrics_util import getUnitArea


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask,
                                   tform, ignoreMask, plot=None):
                     

    if plot is None:
        PLOTS_ENABLE = False
    else:
        PLOTS_ENABLE = True
        PLOTS_SAVE_PREFIX = "thresholdGeometry_"
                                   
    refHgt = (refDSM - refDTM)
    refObj = refHgt
    refObj[~refMask] = 0

    testHgt = (testDSM - testDTM)
    testObj = np.copy(testHgt)
    testObj[~testMask] = 0

    # Make metrics
    refOnlyMask = refMask & ~testMask
    testOnlyMask = testMask & ~refMask
    overlapMask = refMask & testMask

    # Apply ignore mask
    refOnlyMask = refOnlyMask & ~ignoreMask
    testOnlyMask = testOnlyMask & ~ignoreMask
    overlapMask = overlapMask & ~ignoreMask

    
    if PLOTS_ENABLE:
        plot.make(refMask, 'Reference Object Regions', 211, saveName=PLOTS_SAVE_PREFIX+"refObjMask")
        plot.make(refObj,  'Reference Object Height', 212, saveName=PLOTS_SAVE_PREFIX+"refObjHgt", colorbar=True)
        
        plot.make(testMask, 'Test Object Regions', 251, saveName=PLOTS_SAVE_PREFIX+"testObjHgt")
        plot.make(testObj, 'Test Object  Height', 252, saveName=PLOTS_SAVE_PREFIX+"testObjHgt", colorbar=True)
    
        plot.make(refOnlyMask,  'False Negative Regions', 281, saveName=PLOTS_SAVE_PREFIX+"falseNegetive")
        plot.make(testOnlyMask, 'False Positive Regions', 282, saveName=PLOTS_SAVE_PREFIX+"falsePositive")
        plot.make(overlapMask,  'True Positive Regions',  283, saveName=PLOTS_SAVE_PREFIX+"truePositive")
    
    
    
    # Determine evaluation units.
    unitArea = getUnitArea(tform)

    # --- Hard Error ------------------------------------------------------
    # Regions that are 2D False Positives or False Negatives, are
    # all or nothing.  These regions don't consider overlap in the
    # underlying terrain models

    # -------- False Positive ---------------------------------------------
    unitCountFP = np.sum(testOnlyMask)
    oobFP = np.sum(testOnlyMask * testObj) * unitArea

    # -------- False Negative ---------------------------------------------
    unitCountFN = np.sum(refOnlyMask)
    oobFN = np.sum(refOnlyMask * refObj) * unitArea

    # --- Soft Error ------------------------------------------------------
    # Regions that are 2D True Positive

    # For both below:
    #       Positive values are False Positives
    #       Negative values are False Negatives
    deltaTop = testDSM - refDSM
    deltaBot = refDTM - testDTM

    # Regions that are 2D True Positives
    unitCountTP = np.sum(overlapMask)
    overlap = overlapMask * (testObj - refObj)
    overlap[np.isnan(overlap)] = 0

    # -------- False Positive -------------------------------------------------
    false_positives = np.nansum((deltaTop > 0) * deltaTop * overlapMask) * unitArea + \
         np.nansum((deltaBot > 0) * deltaBot * overlapMask) * unitArea

    # -------- False Negative -------------------------------------------------
    false_negatives = -np.nansum((deltaTop < 0) * deltaTop * overlapMask) * unitArea + \
         -np.nansum((deltaBot < 0) * deltaBot * overlapMask) * unitArea

    # -------- True Positive ---------------------------------------------------
    true_positives = np.nansum(refObj * overlapMask) * unitArea - false_negatives
    tolFP = false_positives + oobFP
    tolFN = false_negatives + oobFN
    tolTP = true_positives

    metrics = {
        '2D': calcMops(unitCountTP, unitCountFN, unitCountFP),
        '3D': calcMops(tolTP, tolFN, tolFP),
    }

    if PLOTS_ENABLE:
        errorMap = np.empty(refOnlyMask.shape)
        errorMap[:] = np.nan
        errorMap[testOnlyMask == 1] =  testObj[testOnlyMask == 1]
        errorMap[refOnlyMask == 1]  = -refObj[refOnlyMask == 1]

        overlap = overlapMask * (testDSM - refDSM)
        errorMap[overlapMask == 1]  =  overlap[overlapMask == 1]

        plot.make(errorMap, 'Height Error', 291, saveName=PLOTS_SAVE_PREFIX+"errHgt", colorbar=True)

        errorMap[errorMap > 5] = 5
        errorMap[errorMap < -5] = -5
        plot.make(errorMap, 'Height Error', 292, saveName=PLOTS_SAVE_PREFIX+"errHgtClipped", colorbar=True)

        tmp = deltaTop
        tmp[ignoreMask] = np.nan
        plot.make(tmp, 'DSM Error', 293, saveName=PLOTS_SAVE_PREFIX+"errHgtDSM", colorbar=True)

        tmp = deltaBot
        tmp[ignoreMask] = np.nan
        plot.make(tmp, 'DTM Error', 294, saveName=PLOTS_SAVE_PREFIX+"errHgtDTM", colorbar=True)


    return metrics
