import numpy as np


def calcMops(true_positives, false_negatives, false_positives):
    s = {
        'recall': true_positives / (true_positives + false_negatives),
        'precision': true_positives / (true_positives + false_positives),
        'jaccardIndex': true_positives / (true_positives + false_negatives + false_positives),
        'branchingFactor': false_positives / true_positives,
        'missFactor': false_negatives / true_positives,
    }
    s['completeness'] = s['recall']
    s['correctness'] = s['precision']
    s['fscore'] = (2 * s['recall'] * s['precision']) / (s['recall'] + s['precision'])
    return s


def printMetrics(metrics, results_file_name=None):
    write_to_file = False
    f = None
    if results_file_name:
        write_to_file = True
        f = open(results_file_name, 'w')
        print("Writing metric results to:  %s" % results_file_name)

    def tee(message, include_in_file, file):
        print(message)
        if include_in_file:
            file.writelines("%s\n" % message)

    string = ("%s: %0.3f" % ('Completeness', metrics['completeness']))
    tee(string, write_to_file, f)
    string = ("%s: %0.3f" % ('Correctness', metrics['correctness']))
    tee(string, write_to_file, f)
    string = ("%s: %0.3f" % ('F-Score', metrics['fscore']))
    tee(string, write_to_file, f)
    string = ("%s: %0.3f" % ('Jaccard Index', metrics['jaccardIndex']))
    tee(string, write_to_file, f)
    string = ("%s: %0.3f" % ('Branching Factor', metrics['branchingFactor']))
    tee(string, write_to_file, f)
    string = ("%s: %0.3f" % ('Miss Factor', metrics['missFactor']))
    tee(string, write_to_file, f)

    if 'offset' in metrics:
        val = metrics["offset"]
        string = ("%s: (%0.3f, %0.3f, %0.3f)" % ("Registration Offset (m)", val[0], val[1], val[2]))
        tee(string, write_to_file, f)

    if write_to_file:
        f.close()


def run_threshold_geometry_metrics(refDSM, refDTM, refMask, REF_CLS_VALUE, testDSM, testDTM, testMask, TEST_CLS_VALUE,
                                   tform, xyzOffset, testDSMFilename, ignoreMask):
    refMask = (refMask == REF_CLS_VALUE)
    refHgt = (refDSM - refDTM)
    refObj = refHgt
    refObj[~refMask] = 0

    testMask = (testMask == TEST_CLS_VALUE)
    testHgt = (testDSM - testDTM)
    testObj = np.copy(testHgt)
    testObj[~testMask] = 0

    # Make metrics
    refOnlyMask = refMask & ~testMask
    testOnlyMask = testMask & ~refMask
    overlapMask = refMask & testMask

    # Apply ignore mask
    refOnlyMask = refOnlyMask & ~ignoreMask
    testOnlyMask = testOnlyMask & ~ignoreMask
    overlapMask = overlapMask & ~ignoreMask

    # Determine evaluation units.
    unitArea = abs(tform[1] * tform[5])

    # --- Hard Error ------------------------------------------------------
    # Regions that are 2D False Positives or False Negatives, are
    # all or nothing.  These regions don't consider overlap in the
    # underlying terrain models

    # -------- False Positive ---------------------------------------------
    unitCountFP = np.sum(testOnlyMask)
    oobFP = np.sum(testOnlyMask * testObj) * unitArea

    # -------- False Negative ---------------------------------------------
    unitCountFN = np.sum(refOnlyMask)
    oobFN = np.sum(refOnlyMask * refObj) * unitArea

    # --- Soft Error ------------------------------------------------------
    # Regions that are 2D True Positive

    # For both below:
    #       Positive values are False Positives
    #       Negative values are False Negatives
    deltaTop = testDSM - refDSM
    deltaBot = refDTM - testDTM

    # Regions that are 2D True Positives
    unitCountTP = np.sum(overlapMask)
    overlap = overlapMask * (testObj - refObj)
    overlap[np.isnan(overlap)] = 0

    # -------- False Positive -------------------------------------------------
    false_positives = np.nansum((deltaTop > 0) * deltaTop * overlapMask) * unitArea + \
         np.nansum((deltaBot > 0) * deltaBot * overlapMask) * unitArea

    # -------- False Negative -------------------------------------------------
    false_negatives = -np.nansum((deltaTop < 0) * deltaTop * overlapMask) * unitArea + \
         -np.nansum((deltaBot < 0) * deltaBot * overlapMask) * unitArea

    # -------- True Positive ---------------------------------------------------
    true_positives = np.nansum(refObj * overlapMask) * unitArea - false_negatives
    tolFP = false_positives + oobFP
    tolFN = false_negatives + oobFN
    tolTP = true_positives

    metrics_2d = calcMops(unitCountTP, unitCountFN, unitCountFP)
    metrics_3d = calcMops(tolTP, tolFN, tolFP)

    metrics_3d['offset'] = xyzOffset

    print('')
    print('2D Metrics:')
    results_filename = testDSMFilename + "_2d_metrics.txt"
    printMetrics(metrics_2d, results_filename)

    print('')
    print('3D Metrics:')
    results_filename = testDSMFilename + "_3d_metrics.txt"
    printMetrics(metrics_3d, results_filename)
