import numpy as np
import os

from .metrics_util import calcMops

def run_dtm_accuracy_metrics(refDTM, testDTM, threshold, unitArea, plot=None):

    # Compute Z-RMS Error
    delta = testDTM - refDTM
    zrmse = np.sqrt(np.sum(delta*delta)/delta.size)

    # Compute a correctness and completeness
    TP = np.abs(delta) <= threshold
    FP = delta > threshold
    FN = delta < threshold

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