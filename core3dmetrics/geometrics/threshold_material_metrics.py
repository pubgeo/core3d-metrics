#
# Run CORE3D threshold material labeling metrics and report results.
# This is called by run_core3d_metrics.py
#

import numpy as np
from collections import defaultdict

# Define structure information data structure.
class Structure:
    def __init__(self):
        self.pixels = []  # list of pixel coordinate (x,y) tuples
        self.truthPrimaryMaterial = 0  # index of truth primary structure material
        self.testPrimaryMaterial = 0  # index of test primary structure material


# Return dictionary of structures identified by their indices.
def getStructures(img):
    structuresDic = defaultdict(Structure)
    for y in range(len(img)):
        for x in range(len(img[y])):
            val = img[y][x]
            if val > 0:
                structuresDic[val].pixels.append((x, y))  # add pixel to list for this structure index
    return structuresDic


# Determine the most abundant material index within a structure footprint
#     Returns -1 if no valid material present
def getMaterialFromStructurePixels(img, pixels, materialIndicesToIgnore):
    # Count pixels of each material
    indexCounts = defaultdict(int)
    for p in range(len(pixels)):
        indexCounts[img[pixels[p][1]][pixels[p][0]]] += 1
    # Find most abundant material
    maxMaterialCount = -1
    maxMaterialCountIndex = -1
    for k in indexCounts.keys():
        if indexCounts[k] > maxMaterialCount and k not in materialIndicesToIgnore:
            maxMaterialCount = indexCounts[k]
            maxMaterialCountIndex = k
    return maxMaterialCountIndex

# Moves values from Asph/Con Uncertain class to appropriate asphalt/concrete classes
def mergeConfusionMatrixUncertainAsphaltConcreteCells(confMatrix):
    sz = confMatrix.shape[0]
    confMatrix[1][1] += confMatrix[14][1]
    confMatrix[14][1] = 0
    confMatrix[2][2] += confMatrix[14][2]
    confMatrix[14][2] = 0


def material_plot(refMTL, testMTL, plot):

    PLOTS_SAVE_PREFIX = "thresholdMaterials_"

    # This plot assumes material labels/indices specified in the config file are the same as defined here
    cmap = [
            [0.00,    0.00,    0.00],
            [0.55,    0.55,    0.55],
            [0.20,    0.55,    0.65],
            [1.00,    1.00,    0.11],
            [0.03,    0.40,    0.03],
            [0.47,    0.63,    0.27],
            [0.86,    0.30,    0.10],
            [0.90,    0.00,    0.00],
            [0.31,    0.16,    0.04],
            [0.12,    1.00,    1.78],
            [0.00,    0.00,    1.00],
            [1.00,    1.00,    1.00],
            [1.00,    0.00,    1.00],
            [1.00,    0.39,    1.00],
            [1.00,    0.66,    1.00]
            ]

    labels = [
        "Unclassified",
        "Asphalt",
        "Concrete/Stone",
        "Glass",
        "Tree",
        "Non-tree veg",
        "Metal",
        "Ceramic",
        "Soil",
        "Solar panel",
        "Water",
        "Polymer",
        "Unscored",
        "Indeterminate",
        "Indeterminate Asphalt/Concrete"
         ]

    ticks = list(np.arange(0,len(labels)))


    plot.make(refMTL, 'Reference Materials ', 340, saveName=PLOTS_SAVE_PREFIX + "ref", colorbar=True,
              cmap=cmap, cm_labels=labels, cm_ticks=ticks, vmin=-0.5, vmax=len(labels)-0.5)

    plot.make(testMTL, 'Test Materials', 340, saveName=PLOTS_SAVE_PREFIX + "test", colorbar=True,
              cmap=cmap, cm_labels=labels, cm_ticks=ticks, vmin=-0.5, vmax=len(labels)-0.5)

# Run material labeling metrics and report results.
def run_material_metrics(refNDX, refMTL, testMTL, materialNames, materialIndicesToIgnore, plot=None, verbose=True):
    print("Defined materials:",', '.join(materialNames))
    print("Ignored materials in truth: ",', '.join([materialNames[x] for x in materialIndicesToIgnore]))

    print("Building dictionary of reference structure locations and labels...")
    structuresDic = getStructures(refNDX)
    print("There are ", len(structuresDic), "reference structures.")

    print("Selecting the most abundant material for each structure in reference model...")
    for k in structuresDic.keys():
        maxIdx = getMaterialFromStructurePixels(refMTL, structuresDic[k].pixels, materialIndicesToIgnore)
        structuresDic[k].truthPrimaryMaterial = maxIdx

    print("Selecting the most abundant material for each structure in test model...")
    for k in structuresDic.keys():
        maxIdx = getMaterialFromStructurePixels(testMTL, structuresDic[k].pixels, materialIndicesToIgnore)
        structuresDic[k].testPrimaryMaterial = maxIdx

    # Create pixel label confusion matrix
    np.set_printoptions(linewidth=120)
    pixelConfMatrix = np.zeros((len(materialNames), len(materialNames)), dtype=np.int32)
    for y in range(len(refMTL)):
        for x in range(len(refMTL[y])):
            if refNDX[y][x] != 0: # Limit evaluation to inside structure outlines
                if refMTL[y][x] not in materialIndicesToIgnore: # Limit evaluation to valid materials
                    pixelConfMatrix[refMTL[y][x]][testMTL[y][x]] += 1

    # Re-classify indeterminate asphalt-concrete in pixel-wise confusion matrix
    mergeConfusionMatrixUncertainAsphaltConcreteCells(pixelConfMatrix)

    # Classes present in the reference model (IOU)
    presentRefClasses = pixelConfMatrix.sum(axis=1) > 0

    # Don't include 'Indeterminate asphalt/concrete' in mean IOU, values get resigned to asphalt or concrete
    if 'Indeterminate asphalt/concrete'  in materialNames:
        presentRefClasses[materialNames.index('Indeterminate asphalt/concrete')] = False


    # Compute pixelwise intersection over union
    pixelIOU = np.divide(np.diag(pixelConfMatrix),
                         (pixelConfMatrix.sum(axis=0) + pixelConfMatrix.sum(axis=1) - np.diag(pixelConfMatrix)),
                         out=-np.ones( (1, len(pixelConfMatrix[0])), np.double),
                         where=presentRefClasses!=0)

    # Mean IOU
    pixelMeanIOU = np.mean(pixelIOU[0][presentRefClasses])

    # Dictionary of IOU for each reference matrerial for output
    pixelIOUkvp = dict()
    for x, y in enumerate(np.flatnonzero(presentRefClasses)):
        pixelIOUkvp[materialNames[y].strip()] = pixelIOU[0][y]

    # parse plot input
    if plot is not None:
        material_plot(refMTL, testMTL, plot)

    # Print pixel statistics
    print()
    scoredPixelsCount = np.sum(pixelConfMatrix)
    correctPixelsCount = np.trace(pixelConfMatrix)
    correctPixelsFraction = correctPixelsCount / scoredPixelsCount
    print("Pixel material confusion matrix:")
    print(pixelConfMatrix)
    print("Pixel material IOU:")
    print(pixelIOU)
    print("Pixel material mIOU:", pixelMeanIOU)
    print('Pixelwise IOU by Class:')
    for x in pixelIOUkvp:
        print('', x, ': ', pixelIOUkvp[x])
    print("Total pixels scored: ", scoredPixelsCount)
    print("Total pixels correctly classified: ", correctPixelsCount)
    print("Percent pixels correctly classified: ", str(correctPixelsFraction * 100) + "%")
    print()

    # Create structure label confusion matrix
    unscoredCount = 0
    structureConfMatrix = np.zeros((len(materialNames), len(materialNames)), dtype = np.int32)
    for k in structuresDic.keys():
        if structuresDic[k].truthPrimaryMaterial not in materialIndicesToIgnore and structuresDic[k].truthPrimaryMaterial != -1:
            structureConfMatrix[structuresDic[k].truthPrimaryMaterial][structuresDic[k].testPrimaryMaterial] += 1
        else:
            unscoredCount += 1

    # Re-classify indeterminate asphalt-concrete in structure confusion matrix
    mergeConfusionMatrixUncertainAsphaltConcreteCells(structureConfMatrix)

    # Print structure statistics
    scoredStructuresCount = np.sum(structureConfMatrix)
    correctStructuresCount = np.trace(structureConfMatrix)
    correctStructuresFraction = correctStructuresCount / scoredStructuresCount
    print("Primary structure material confusion matrix:")
    print(structureConfMatrix)
    print("Structures marked as non-scored: ", unscoredCount)
    print("Total structures scored: ", scoredStructuresCount)
    print("Total structures correctly classified: ", correctStructuresCount)
    print("Percent structures correctly classified: ", str(correctStructuresFraction * 100) + "%")

    metrics = {
        'scored_structures': int(scoredStructuresCount),
        'fraction_structures_correct': correctStructuresFraction,
        'fraction_pixels_correct': correctPixelsFraction,
        'structurewise_confusion_matrix': str(structureConfMatrix),
        'pixelwise_mIOU': pixelMeanIOU,
        'pixelwise_IOU': pixelIOUkvp,
		'pixelwise_confusion_matrix': str(pixelConfMatrix)
    }
	
    return metrics
	

