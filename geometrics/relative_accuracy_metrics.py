
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
        plot.make(delta, 'Terrain Model - Height Error', 581, saveName="relVertAcc_hgtErr", colorbar=True)


    metrics = {
        'zrmse': zrmse
    }

    return metrics