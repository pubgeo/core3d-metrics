import numpy as np
import os
import sys

from .metrics_util import calcMops

def run_terrain_accuracy_metrics(refDSM, refDTM, testDSM, testDTM, refMask, testMask, threshold=1, unitArea=1, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # Compute Z error percentiles.
    # Ignore objects identified by the reference mask because the
    # ground is not expected to be observable under those objects.
    delta = testDTM - refDTM
    delta_minus_mask = delta[np.where(refMask == 0)]
    z68 = np.percentile(abs(delta_minus_mask),68)
    z50 = np.percentile(abs(delta_minus_mask),50)
    z90 = np.percentile(abs(delta_minus_mask),90)

    # Determine ground and not-ground using threshold distance between DSM and DTM.
    # It would be more accurate to define this explicitly in the reference 
    # and test classification labels, but we currently don't require ground to be
    # explicitly labeled in the input files. We should consider changing this, but
    # this is a close approximation for now.
    groundTruth = abs(refDSM - refDTM) < threshold
    groundTest = abs(testDSM - testDTM) < threshold

    # Compute correctness and completeness
    TP = groundTruth & groundTest;
    FP = (groundTruth == 0) & groundTest;
    FN = groundTruth & (groundTest == 0);

    if PLOTS_ENABLE:
        plot.make(delta, 'Terrain Model - Height Error', 481, saveName="terrainAcc_HgtErr", colorbar=True)

        plot.make(TP, 'Terrain Model - True Positive', 482, saveName="terrainAcc_truePositive")
        plot.make(FP, 'Terrain Model - False Positive', 483, saveName="terrainAcc_falsePositive")
        plot.make(FN, 'Terrain Model - False Negetive', 484, saveName="terrainAcc_falseNegetive")

        errorMap = TP*delta
        errorMap[TP == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - True Positive - Height Error', 492, saveName='terrainAcc_truePositive_HgtErr', colorbar=True)

        errorMap = FP*delta
        errorMap[FP == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - False Positive - Height Error', 493, saveName='terrainAcc_falsePositive_HgtErr', colorbar=True)

        errorMap = FN*delta
        errorMap[FN == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - False Negetive - Height Error', 494, saveName='terrainAcc_falseNegetive_HgtErr', colorbar=True)

    # Count number of pixels for 2D metrics
    unitCountTP = np.sum(TP)
    unitCountFP = np.sum(FP)
    unitCountFN = np.sum(FN)

    metrics = {
        'z50': z50,
        'zrmse': z68,
        'z90': z90,
        '2D': calcMops(unitCountTP, unitCountFN, unitCountFP)
    }

    return metrics