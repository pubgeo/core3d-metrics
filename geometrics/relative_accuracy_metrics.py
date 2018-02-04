
import numpy as np

def run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # Evaluate only in overlap region
    evalMask = refMask & testMask

    # Compute Z-RMS Error
    delta = testDSM - refDSM
    delta = delta*evalMask
    zrmse = np.sqrt(np.sum(delta * delta) / delta.size)

    if PLOTS_ENABLE:
        errorMap = delta;
        delta[evalMask == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - Height Error', 581, saveName="relVertAcc_hgtErr", colorbar=True)

        errorMap[errorMap > 5] = 5
        errorMap[errorMap < -5] = -5
        plot.make(errorMap, 'Terrain Model - Height Error', 582, saveName="relVertAcc_hgtErr_clipped", colorbar=True)




    metrics = {
        'zrmse': zrmse
    }

    return metrics