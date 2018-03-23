
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

def run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask, gsd, plot=None):

    PLOTS_ENABLE = True
    if plot is None: PLOTS_ENABLE = False

    # valid mask (opposite of ignore mask)
    validMask = ~ignoreMask

    # Compute relative vertical accuracy
    # Consider only objects selected in both reference and test masks.

    # Calculate Z percentile errors.
    # Z68 approximates ZRMSE assuming normal error distribution.
    delta = testDSM - refDSM
    overlap = refMask & testMask & validMask
    z68 = np.percentile(abs(delta[overlap]),68)
    z50 = np.percentile(abs(delta[overlap]),50)
    z90 = np.percentile(abs(delta[overlap]),90)

    # Generate relative vertical accuracy plots
    if PLOTS_ENABLE:
        errorMap = np.copy(delta)
        errorMap[~overlap] = np.nan
        plot.make(errorMap, 'Object Height Error', 581, saveName="relVertAcc_hgtErr", colorbar=True)
        plot.make(errorMap, 'Object Height Error (Clipped)', 582, saveName="relVertAcc_hgtErr_clipped", colorbar=True,
            vmin=-5,vmax=5)

    # Compute relative horizontal accuracy
    # Consider only objects selected in reference mask.

    # Find region edge pixels
    kernel = np.ones((3, 3), np.int)
    refEdge = convolve2d(refMask.astype(np.int), kernel, mode="same", boundary="symm")
    testEdge = convolve2d(testMask.astype(np.int), kernel, mode="same", boundary="symm")
    validEdge = convolve2d(validMask.astype(np.int), kernel, mode="same", boundary="symm")
    refEdge = (refEdge < 9) & refMask & (validEdge == 9) 
    testEdge = (testEdge < 9) & testMask & (validEdge == 9)
    refPts = refEdge.nonzero()
    testPts = testEdge.nonzero()

    # Use KD Tree to find test point nearest each reference point
    tree = cKDTree(np.transpose(testPts))
    dist, indexes = tree.query(np.transpose(refPts))
    dist = dist * gsd

    # Calculate horizontal percentile errors.
    # H63 approximates HRMSE assuming binormal error distribution.
    h63 = np.percentile(abs(dist),63)
    h50 = np.percentile(abs(dist),50)
    h90 = np.percentile(abs(dist),90)

    # Generate relative horizontal accuracy plots
    if PLOTS_ENABLE:
        plot.make(refEdge, 'Reference Model Perimeters', 591,
                  saveName="relHorzAcc_edgeMapRef", cmap='Greys')
        plot.make(testEdge, 'Test Model Perimeters', 592,
                  saveName="relHorzAcc_edgeMapTest", cmap='Greys')

        plt = plot.make(None,'Relative Horizontal Accuracy')
        plt.imshow(refMask & validMask, cmap='Greys')
        plt.plot(refPts[1], refPts[0], 'r,')
        plt.plot(testPts[1], testPts[0], 'b,')

        plt.plot((refPts[1], testPts[1][indexes]), (refPts[0], testPts[0][indexes]), 'y', linewidth=0.05)
        plot.save("relHorzAcc_nearestPoints")

    metrics = {
        'z50': z50,
        'zrmse': z68,
        'z90': z90,
		'h50': h50,
        'hrmse': h63,
        'h90': h90
    }
    return metrics
