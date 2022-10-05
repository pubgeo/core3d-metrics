import numpy as np
import os
import sys

from .metrics_util import calcMops

def run_terrain_accuracy_metrics(refDTM, testDTM, refMask, threshold=1, no_data_value = -10000, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # Compute Z error percentiles.
    # Ignore objects identified by the reference mask because the
    # ground is not expected to be observable under those objects.
    delta = testDTM - refDTM
    # Account for no data
    refMask[refDTM == no_data_value] = 1
    # Get delta mask
    delta_minus_mask = delta[np.where(refMask == 0)]
    z68 = np.percentile(abs(delta_minus_mask),68)
    z50 = np.percentile(abs(delta_minus_mask),50)
    z90 = np.percentile(abs(delta_minus_mask),90)

    # Compute DTM completeness.
    match = abs(testDTM - refDTM) < threshold
    completeness = np.sum(match)/np.size(match)

    # This is a hack to avoid water flattening at z = -1 in reference DTM files from 133 US Cities 
    # from corrupting the results. The water should be properly labeled instead, but this
    # is unlikely to cause problems for our test areas. Just keep an eye on it.
    distanceFromWater = abs(refDTM + 1.0)
    match = match[np.where(distanceFromWater > 0.2)]
    completeness_water_removed = np.sum(match)/np.size(match)
	
    if PLOTS_ENABLE:
        errorMap = delta
        errorMap[refMask] = np.nan
        errorMap[errorMap > 5] = 5
        errorMap[errorMap < -5] = -5
        plot.make(delta, 'Terrain Model - Height Error', 481, saveName="terrainAcc_HgtErr", colorbar=True)

    metrics = {
        'z50': z50,
        'zrmse': z68,
        'z90': z90,
		'completeness': completeness,
		'completeness_water_removed': completeness_water_removed
    }

    return metrics