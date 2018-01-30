import numpy as np
import os


def calcMops(true_positives, false_negatives, false_positives):
    s = {
        'recall': true_positives / (true_positives + false_negatives),
        'precision': true_positives / (true_positives + false_positives),
        'jaccardIndex': true_positives / (true_positives + false_negatives + false_positives),
        'branchingFactor': false_positives / true_positives,
        'missFactor': false_negatives / true_positives,
    }
    s['completeness'] = s['recall']
    s['correctness'] = s['precision']
    s['fscore'] = (2 * s['recall'] * s['precision']) / (s['recall'] + s['precision'])
    return s


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, REF_CLS_VALUE, testDSM, testDTM, testMask, TEST_CLS_VALUE,
                                   tform, ignoreMask, plot=None):
                     
    PLOTS_ENABLE = True                        
    if plot is None: PLOTS_ENABLE = Talse
                                   
    refMask = (refMask == REF_CLS_VALUE)
    refHgt = (refDSM - refDTM)
    refObj = refHgt
    refObj[~refMask] = 0

    testMask = (testMask == TEST_CLS_VALUE)
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
        plot.make(refMask, 'Reference Object Mask', 211, saveName="refMask")
        plot.make(refObj,  'refObj', 212, colorbar=True)
        
        plot.make(testMask, 'Test Object Mask', 251, saveName="testMask")
        plot.make(testObj, 'testObj', 252, colorbar=True)
    
        plot.make(refOnlyMask,  'False Negative Mask', 281, saveName='FN')
        plot.make(testOnlyMask, 'False Positive Mask', 282, saveName='FP')
        plot.make(overlapMask,  'True Positive Mask',  283, saveName='TP')
    
    
    
    # Determine evaluation units.
    unitArea = abs(tform[1] * tform[5])

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

        plot.make(errorMap, '3D Error', 291, saveName='Err3D', colorbar=True)

        errorMap[errorMap > 5] = 5
        errorMap[errorMap < -5] = -5
        plot.make(errorMap, '3D Error', 292, saveName='Err3D_Clipped', colorbar=True)

        tmp = deltaTop
        tmp[ignoreMask] = np.nan
        plot.make(tmp, 'DSM Error', 293, saveName='ErrDSM', colorbar=True)

        tmp = deltaTop
        tmp[ignoreMask] = np.nan
        plot.make(tmp, 'DTM Error', 294, saveName='ErrDTM', colorbar=True)
    
    
    return metrics
