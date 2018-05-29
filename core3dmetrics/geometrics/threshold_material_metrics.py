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

# Merges class1 into class2
def mergeConfusionMatrixClasses(confMatrix, class1, class2):
    sz = confMatrix.shape[0]
    for i in range(0, sz):
        confMatrix[class2][i] += confMatrix[class1][i]
        confMatrix[class1][i] = 0
    for i in range(0, sz):
        confMatrix[i][class2] += confMatrix[i][class1]
        confMatrix[i][class1] = 0


# Run material labeling metrics and report results.
def run_material_metrics(refNDX, refMTL, testMTL, materialNames, materialIndicesToIgnore):
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

    # Merge asphalt-concrete in pixel-wise confusion matrix
    mergeConfusionMatrixClasses(pixelConfMatrix, 2, 1)

    # Print pixel statistics
    print()
    scoredPixelsCount = np.sum(pixelConfMatrix)
    correctPixelsCount = np.trace(pixelConfMatrix)
    correctPixelsFraction = correctPixelsCount / scoredPixelsCount
    print("Pixel material confusion matrix:")
    print(pixelConfMatrix)
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

    # Merge asphalt-concrete in building confusion matrix
    mergeConfusionMatrixClasses(structureConfMatrix, 2, 1)

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
        'fraction_pixels_correct': correctPixelsFraction
    }
	
    return metrics
	

