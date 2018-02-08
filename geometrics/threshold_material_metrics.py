#
# Run CORE3D threshold material labeling metrics and report results.
# This is called by run_core3d_metrics.py
#

import numpy as np
from collections import defaultdict

# Define building information data structure.
class Building:
    def __init__(self):
        self.pixels = []  # list of pixel coordinate (x,y) tuples
        self.truthPrimaryMaterial = 0  # index of truth primary building material
        self.testPrimaryMaterial = 0  # index of test primary building material


# Return dictionary of buildings identified by their indices.
def getBuildings(img):
    buildingsDic = defaultdict(Building)
    for y in range(len(img)):
        for x in range(len(img[y])):
            val = img[y][x]
            if val > 0:
                buildingsDic[val].pixels.append((x, y))  # add pixel to list for this building index

    # Remove very small buildings.
    # TODO: Determine why this is being done. Seems like we should remove it.
    for k in list(buildingsDic.keys()):
        if len(buildingsDic[k].pixels) < 10:
            del buildingsDic[k]
    return buildingsDic


# Determine the most abundant material index within a building footprint.
def getMaterialFromBuildingPixels(img, pixels):
    indexCounts = defaultdict(int)
    for p in range(len(pixels)):
        indexCounts[img[pixels[p][1]][pixels[p][0]]] += 1
    maxMaterialCount = -1
    for k in indexCounts.keys():
        maxMaterialCount = max(maxMaterialCount, indexCounts[k])
    maxMaterialCountIndex = -1
    for k in indexCounts.keys():
        if indexCounts[k] == maxMaterialCount:
            maxMaterialCountIndex = k
    return maxMaterialCountIndex


# Run material labeling metrics and report results.
def run_material_metrics(refNDX, refMTL, testMTL, materialNames, materialIndicesToIgnore):

    print("Building dictionary of reference building locations and labels...")
    buildingsDic = getBuildings(refNDX)
    print("Found ", len(buildingsDic), "reference buildings.")

    print("Selecting the most abundant material for each building in reference model...")
    for k in buildingsDic.keys():
        maxIdx = getMaterialFromBuildingPixels(refMTL, buildingsDic[k].pixels)
        buildingsDic[k].truthPrimaryMaterial = maxIdx

    print("Selecting the most abundant material for each building in test model...")
    for k in buildingsDic.keys():
        maxIdx = getMaterialFromBuildingPixels(testMTL, buildingsDic[k].pixels)
        buildingsDic[k].testPrimaryMaterial = maxIdx

    print("Building material labeling confusion matrix...")
    confMatrix = np.zeros((len(materialNames), len(materialNames)))
    unscoredCount = 0
    for k in buildingsDic.keys():
        if buildingsDic[k].truthPrimaryMaterial not in materialIndicesToIgnore:
            confMatrix[buildingsDic[k].truthPrimaryMaterial][buildingsDic[k].testPrimaryMaterial] += 1
        else:
            unscoredCount += 1

    print("Computing statistics...")
    scoredBuildingsCount = np.sum(confMatrix)
    correctBuildingsCount = np.trace(confMatrix)
    correctBuildingsFraction = correctBuildingsCount / scoredBuildingsCount

    print("Confusion matrix:")
    print(confMatrix)
    print()

    print("Buildings marked as non-scored: ", unscoredCount)
    print("Total buildings scored: ", scoredBuildingsCount)
    print("Total buildings correctly classified: ", correctBuildingsCount)
    print("Percent correctly classified: ", str(correctBuildingsFraction * 100) + "%")

# TODO: Update to assess separately for different types of object (building, bridge, etc.).
# TODO: Consider penalizing any missing objects in the product as misclassified.
