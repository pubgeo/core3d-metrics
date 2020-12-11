
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis

def run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask, gsd, for_objectwise=False, plot=None):

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
    signed_z_errors = delta[overlap]
    try:
        zrmse_explicit = np.sqrt(sum(signed_z_errors ** 2)/len(signed_z_errors))
    except ZeroDivisionError:
        print("Error")
        zrmse_explicit = np.nan
        signed_z_errors = np.array([np.nan])
    if np.unique(overlap).size is 1:
        z68 = 100
        z50 = 100
        z90 = 100
    else:
        z68 = np.percentile(abs(delta[overlap]), 68)
        z50 = np.percentile(abs(delta[overlap]), 50)
        z90 = np.percentile(abs(delta[overlap]), 90)

    # Generate relative vertical accuracy plots
    PLOTS_ENABLE = False
    if PLOTS_ENABLE:
        errorMap = np.copy(delta)
        errorMap[~overlap] = np.nan
        plot.make(errorMap, 'Object Height Error', 581, saveName="relVertAcc_hgtErr", colorbar=True)
        plot.make(errorMap, 'Object Height Error (Clipped)', 582, saveName="relVertAcc_hgtErr_clipped", colorbar=True,
            vmin=-5, vmax=5)

    # Compute relative horizontal accuracy
    # Consider only objects selected in reference mask.

    # Find region edge pixels
    if np.histogram(refMask)[0][9] != 1:
        kernel = np.ones((3, 3), np.int)
        refEdge = convolve2d(refMask.astype(np.int), kernel, mode="same", boundary="symm")
        testEdge = convolve2d(testMask.astype(np.int), kernel, mode="same", boundary="symm")
        validEdge = convolve2d(validMask.astype(np.int), kernel, mode="same", boundary="symm")
        refEdge = (refEdge < 9) & refMask & (validEdge == 9)
        testEdge = (testEdge < 9) & testMask & (validEdge == 9)
        refPts = refEdge.nonzero()
        testPts = testEdge.nonzero()
        if (np.unique(testEdge).__len__() == 1 and np.unique(testEdge)[0] == False) or \
                (np.unique(refEdge).__len__() == 1 and np.unique(refEdge)[0] == False):
            refPts = np.where(refMask == True)
            testPts = np.where(testMask == True)
    else:
        refPts = np.where(refMask == True)
        testPts = np.where(testMask == True)

    # Use KD Tree to find test point nearest each reference point
    refPts_transpose = np.transpose(refPts)
    testPts_transpose = np.transpose(testPts)
    signed_x_errors = []
    signed_y_errors = []
    try:
        tree = cKDTree(np.transpose(testPts))
        dist, indexes = tree.query(np.transpose(refPts))
        # Store the x and y distances
        for refPt_index, testPt_index in enumerate(indexes):
            refPt = refPts_transpose[refPt_index]
            testPt = testPts_transpose[testPt_index]
            signed_x_errors.append(testPt[1] - refPt[1])
            signed_y_errors.append(testPt[0] - refPt[0])
        signed_x_errors = np.array(signed_x_errors) * gsd
        signed_y_errors = np.array(signed_y_errors) * gsd
        dist = dist * gsd
        # Sanity check
        for i, x in enumerate(signed_x_errors):
            if np.sqrt(signed_x_errors[i] ** 2 + signed_y_errors[i] ** 2) != dist[i]:
                print("Error!")
    except ValueError:
        dist = np.nan

    # Calculate horizontal percentile errors.
    # H63 approximates HRMSE assuming binormal error distribution.
    h63 = np.percentile(abs(dist), 63)
    h50 = np.percentile(abs(dist), 50)
    h90 = np.percentile(abs(dist), 90)
    hrmse_explicit = np.sqrt(sum(dist ** 2)/len(dist))

    # Get statistics for data
    signed_x_errors_skew = skew(signed_x_errors)
    signed_y_errors_skew = skew(signed_y_errors)
    signed_z_errors_skew = skew(signed_z_errors)

    signed_x_errors_kurtosis = kurtosis(signed_x_errors)
    signed_y_errors_kurtosis = kurtosis(signed_y_errors)
    signed_z_errors_kurtosis = kurtosis(signed_z_errors)

    signed_x_errors_median = float(np.median(signed_x_errors).item())
    signed_y_errors_median = float(np.median(signed_y_errors).item())
    signed_z_errors_median = float(np.median(signed_z_errors).item())

    signed_x_errors_var = float(np.var(signed_x_errors).item())
    signed_y_errors_var = float(np.var(signed_y_errors).item())
    signed_z_errors_var = float(np.var(signed_z_errors).item())

    signed_x_errors_mean = float(np.mean(signed_x_errors).item())
    signed_y_errors_mean = float(np.mean(signed_y_errors).item())
    signed_z_errors_mean = float(np.mean(signed_z_errors).item())

    bin_range_horz = 1.0  # meters
    bin_range_vert = 0.5  # meters
    number_of_bins_x = int(np.ceil(abs(signed_x_errors.max() - signed_x_errors.min())/bin_range_horz))
    number_of_bins_y = int(np.ceil(abs(signed_y_errors.max() - signed_y_errors.min()) / bin_range_horz))
    try:
        number_of_bins_z = int(np.ceil(abs(signed_z_errors.max() - signed_z_errors.min())/bin_range_vert))
    except ValueError:
        number_of_bins_z = np.nan
    # Generate histogram
    if not for_objectwise:
        plot.make_distance_histogram(signed_x_errors, fig=593, plot_title='Signed X Errors', bin_width=bin_range_horz, bins=number_of_bins_x)
        plot.make_distance_histogram(signed_y_errors, fig=594, plot_title='Signed Y Errors', bin_width=bin_range_horz, bins=number_of_bins_y)
        try:
            plot.make_distance_histogram(signed_z_errors, fig=595, plot_title='Signed Z Errors', bin_width=bin_range_vert, bins=number_of_bins_z)
        except Exception:
            print("Couldn't make z error plot. Something went wrong...")

    # Generate relative horizontal accuracy plots
    PLOTS_ENABLE = False  # Turn off this feature unless otherwise because it takes a lot of time
    if PLOTS_ENABLE:
        plot.make(refEdge, 'Reference Model Perimeters', 591,
                  saveName="relHorzAcc_edgeMapRef", cmap='Greys')
        plot.make(testEdge, 'Test Model Perimeters', 592,
                  saveName="relHorzAcc_edgeMapTest", cmap='Greys')

        plt = plot.make(None,'Relative Horizontal Accuracy')
        plt.imshow(refMask & validMask, cmap='Greys')
        plt.plot(refPts[1], refPts[0], 'r,')
        plt.plot(testPts[1], testPts[0], 'b,')
        try:
            plt.plot((refPts[1], testPts[1][indexes]), (refPts[0], testPts[0][indexes]), 'y', linewidth=0.05)
            plot.save("relHorzAcc_nearestPoints")
        except NameError:
            # Not possible to calculate HRMSE because lack of performer polygon, namely indexes variable
            pass

    metrics = {
        'z50': z50,
        'zrmse': z68,
        'z90': z90,
        'h50': h50,
        'hrmse': h63,
        'h90': h90,
        'zrmse_explicit': zrmse_explicit,
        'hrmse_explicit': hrmse_explicit,
        'signed_x_errors_kurtosis': signed_x_errors_kurtosis,
        'signed_y_errors_kurtosis': signed_y_errors_kurtosis,
        'signed_z_errors_kurtosis': signed_z_errors_kurtosis,
        'signed_x_errors_skew': signed_x_errors_skew,
        'signed_y_errors_skew': signed_y_errors_skew,
        'signed_z_errors_skew': signed_z_errors_skew,
        'signed_x_errors_median': signed_x_errors_median,
        'signed_y_errors_median': signed_y_errors_median,
        'signed_z_errors_median': signed_z_errors_median,
        'signed_x_errors_var': signed_x_errors_var,
        'signed_y_errors_var': signed_y_errors_var,
        'signed_z_errors_var': signed_z_errors_var,
        'signed_x_errors_mean': signed_x_errors_mean,
        'signed_y_errors_mean': signed_y_errors_mean,
        'signed_z_errors_mean': signed_z_errors_mean
    }
    return metrics
