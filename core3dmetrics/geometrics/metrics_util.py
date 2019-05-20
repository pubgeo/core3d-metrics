import numpy as np

def calcMops(true_positives, false_negatives, false_positives):

    # when user gets nothing correct
    if (true_positives == 0):
        s = {
            'recall': 0,
            'precision': 0,
            'jaccardIndex': 0,
            'branchingFactor': np.nan,
            'missFactor': np.nan,
            'completeness': 0,
            'correctness': 0,
            'fscore': np.nan
        }

    else:
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

    # append actual TP/FN/FP to report
    s['TP'] = float(true_positives)
    s['FN'] = float(false_negatives)
    s['FP'] = float(false_positives)

    return s


def getUnitArea(tform):
    return abs(tform[1] * tform[5])


def getUnitHeight(tform):
    return (abs(tform[1]) + abs(tform[5])) / 2

def getUnitWidth(tform):
    return (abs(tform[1]) + abs(tform[5])) / 2

# Checks is match values are present as CLS values, and
# expands any special cases
#
# CLS range from 0 to 255 per ASPRS
# (American Society for Photogrammetry and Remote Sensing)
# LiDAR point cloud classification LAS standard
#
# Any special cases are values outside [0 255]
def validateMatchValues(matchValues, clsValues):

    if not isinstance(matchValues, (list)): matchValues = [matchValues]

    outValues = []
    for v in matchValues:

        if v is 256:
            # All non-zero classes
            [outValues.append(c) if c!=0 else None for c in clsValues]
        else:
            # Keep match value only if exists as CLS value
            if v in clsValues: outValues.append(v)

    return outValues

def getMatchValueSets(refCLS_matchSets, testCLS_matchSets, refCLS_classes, testCLS_classes):

    # Classes present in CLS input images
    print("Reference classification values: " + str(refCLS_classes))
    print("Test classification values: " + str(testCLS_classes))

    # Sets of classes specified for evaluation
    if len(refCLS_matchSets) != len(testCLS_matchSets):
        print("WARNING: Inconsistent number of sets specified by CLSMatchValue")
        testCLS_matchSets.clear()
        refCLS_matchSets.clear()

    refCLS_matchSetsValid = []
    testCLS_matchSetsValid = []
    for index, (refMatchValue, testMatchValue) in enumerate(zip(refCLS_matchSets, testCLS_matchSets)):
        refMatchValueValid = validateMatchValues(refMatchValue, refCLS_classes)
        testMatchValueValid = validateMatchValues(testMatchValue, testCLS_classes)

        if len(refMatchValueValid) and len(testMatchValueValid):
            refCLS_matchSetsValid.append(refMatchValueValid)
            testCLS_matchSetsValid.append(testMatchValueValid)


    return refCLS_matchSetsValid, testCLS_matchSetsValid


# Classification Values used are defined by
# ASPRS Standard LIDAR Point Classes 1.4
# http://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf
def clsDecoderRing():

    decoderRing = {
        0: "Never classified",
        1: "Unassigned",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        13: "Wire - Guard(Shield)",
        14: "Wire - Conductor(Phase)",
        15: "Transmission Tower",
        16: "Wire - Structure Connector(Insulator)",
        17: "Bridge Deck",
        18: "High Noise"}

        # 8: Reserved
        # 12: Reserved
        # 19 - 63:  Reserved
        # 64 - 255: User Definable

    return decoderRing
