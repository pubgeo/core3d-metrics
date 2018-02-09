
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

def run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # Compute relative vertical accuracy

    # Evaluate only in overlap region
    evalMask = refMask & testMask

    # Calculate Z-RMS Error
    delta = testDSM - refDSM
    delta = delta*evalMask
    zrmse = np.sqrt(np.sum(delta * delta) / delta.size)

    # Generate relative vertical accuracy plots
    if PLOTS_ENABLE:
        errorMap = delta
        delta[evalMask == 0] = np.nan
        plot.make(errorMap, 'Terrain Model - Height Error', 581, saveName="relVertAcc_hgtErr", colorbar=True)

        errorMap[errorMap > 5] = 5
        errorMap[errorMap < -5] = -5
        plot.make(errorMap, 'Terrain Model - Height Error', 582, saveName="relVertAcc_hgtErr_clipped", colorbar=True)

    # Compute relative horizontal accuracy

    # Find region edge pixels
    kernel = np.ones((3, 3), np.int)
    refEdge = convolve2d(refMask.astype(np.int), kernel, mode="same", boundary="symm")
    testEdge = convolve2d(testMask.astype(np.int), kernel, mode="same", boundary="symm")
    refEdge = (refEdge < 9) & refMask
    testEdge = (testEdge < 9) & testMask
    refPts = refEdge.nonzero()
    testPts = testEdge.nonzero()

    # Use KD Tree to find test point nearest each reference point
    tree = cKDTree(np.transpose(testPts))
    dist, indexes = tree.query(np.transpose(refPts))

    hrmse = np.sqrt(np.sum(dist * dist) / dist.size)

    # Generate relative horizontal accuracy plots
    if PLOTS_ENABLE:
        plot.make(refEdge, 'Reference Model Perimeters', 591,
                  saveName="relHorzAcc_edgeMapRef", cmap='Greys')
        plot.make(testEdge, 'Test Model Perimeters', 592,
                  saveName="relHorzAcc_edgeMapTest", cmap='Greys')

        plt = plot.make(None,'Relative Horizontal Accuracy')
        plt.imshow(refMask, cmap='Greys')
        plt.plot(refPts[1], refPts[0], 'r,')
        plt.plot(testPts[1], testPts[0], 'b,')

        plt.plot((refPts[1], testPts[1][indexes]), (refPts[0], testPts[0][indexes]), 'y', linewidth=0.05)
        plot.save("relHorzAcc_nearestPoints")

    metrics = {
        'zrmse': zrmse,
        'hrmse': hrmse
    }
    return metrics
