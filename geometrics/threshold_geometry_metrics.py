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


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask,
                                   tform, ignoreMask):
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

    return metrics
