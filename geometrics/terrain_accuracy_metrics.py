import numpy as np
import os

import sys

from .metrics_util import calcMops

def run_terrain_accuracy_metrics(refDSM, refDTM, testDSM, testDTM, refMask, testMask, threshold=1, unitArea=1, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # TODO: Should we mask out made-made object? /  Only do ground?
    ignorMask = refMask | testMask

    # Compute Z-RMS Error
    delta = testDTM - refDTM
#    zrmse = np.sqrt(np.sum(delta*delta)/delta.size)
# assume normal distribution and report the 68th percentile
# ignore under buildings
    deltaWithoutBldgs = delta[np.where(refMask == 0)];
    z68 = np.percentile(abs(deltaWithoutBldgs),68);
    z50 = np.percentile(abs(deltaWithoutBldgs),50);
    z90 = np.percentile(abs(deltaWithoutBldgs),90);
	
# also print percentiles for now so we can look at them
#    print('DTM comparisons...');
#    print('90% = ', np.percentile(abs(deltaWithoutBldgs),90));	
#    print('50% = ', np.percentile(abs(deltaWithoutBldgs),50));
#    print('68% = ', np.percentile(abs(deltaWithoutBldgs),68));

	# Define ground/not-ground in reference using threshold distance.
    groundTruth = abs(refDSM - refDTM) < threshold;
    groundTest = abs(testDSM - testDTM) < threshold;
	
    # Compute correctness and completeness
#    TP = np.abs(delta) <= threshold
#    FP = delta > threshold
#    FN = delta < -threshold
    TP = groundTruth & groundTest;
    FP = (groundTruth == 0) & groundTest;
    FN = groundTruth & (groundTest == 0);

    if PLOTS_ENABLE:
        plot.make(delta, 'Terrain Model - Height Error', 481, saveName="dtm_HgtErr", colorbar=True)

        plot.make(TP, 'Terrain Model - True Positive', 482, saveName="dtmTP_Mask")
        plot.make(FP, 'Terrain Model - False Positive', 483, saveName="dtmFP_Mask")
        plot.make(FN, 'Terrain Model - False Negetive', 484, saveName="dtmFN_Mask")

        errorMap = TP*delta
        errorMap[TP == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - True Positive - Height Error', 492, saveName='dtmTP_HgtErr', colorbar=True)

        errorMap = FP*delta
        errorMap[FP == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - False Positive - Height Error', 493, saveName='dtmFP_HgtErr', colorbar=True)

        errorMap = FN*delta
        errorMap[FN == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - False Negetive - Height Error', 494, saveName='dtmFN_HgtErr', colorbar=True)

    # Count number of pixels for 2D metrics
    unitCountTP = np.sum(TP)
    unitCountFP = np.sum(FP)
    unitCountFN = np.sum(FN)

    # Compute positive volumes for 3D metrics
#    delta = np.abs(delta)
#    volumeTP = np.sum(TP * delta) * unitArea
#    volumeFN = np.sum(FP * delta) * unitArea
#    volumeFP = np.sum(FN * delta) * unitArea

    metrics = {
        'z50': z50,
		'z68 (zrmse approximation, assuming normal with zero mean)': z68,
		'z90': z90,
        '2D': calcMops(unitCountTP, unitCountFN, unitCountFP),
 #       '3D': calcMops(volumeTP, volumeFN, volumeFP),
    }

    return metrics