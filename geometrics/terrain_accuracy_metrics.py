import numpy as np
import os

from .metrics_util import calcMops

def run_terrain_accuracy_metrics(refDTM, testDTM, retMask, testMask, threshold=1, unitArea=1, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # TODO: Should we mask out made-made object? /  Only do ground?
    ignorMask = retMask | testMask

    # Compute Z-RMS Error
    delta = testDTM - refDTM
    zrmse = np.sqrt(np.sum(delta*delta)/delta.size)

    # Compute a correctness and completeness
    TP = np.abs(delta) <= threshold
    FP = delta > threshold
    FN = delta < -threshold

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

    # Compute positive volumes for 3D metrics
    delta = np.abs(delta)
    volumeTP = np.sum(TP * delta) * unitArea
    volumeFN = np.sum(FP * delta) * unitArea
    volumeFP = np.sum(FN * delta) * unitArea

    metrics = {
        'zrmse': zrmse,
        '2D': calcMops(unitCountTP, unitCountFN, unitCountFP),
        '3D': calcMops(volumeTP, volumeFN, volumeFP),
    }

    return metrics