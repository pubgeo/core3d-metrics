import numpy as np
from pathlib import Path
import math
import random
import sys
from tqdm import tqdm

try:
    from OrthoImage import OrthoImage
except:
    from utils.OrthoImage import OrthoImage


MAX_FLOAT = sys.float_info.max


class AlignResult:
    def __init__(self):
        self.tx = None
        self.ty = None
        self.tz = None
        self.rms = None


class AlignParameters:
    def __init__(self):
        self.gsd = None
        self.maxdz = None
        self.maxt = None


class AlignBounds:
    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.width = None
        self.height = None


def computeRMS(dx, dy, numSamples, maxSamples, xlist, ylist, referenceDSM: OrthoImage,
               targetDSM: OrthoImage, medianDZ, rms, ndx, completeness) -> bool:
    count = 0
    ndx = 0
    differences = []
    while count < numSamples and ndx < maxSamples:
        x = xlist[ndx]
        y = ylist[ndx]
        ndx += 1

        # Map the point into the target
        # Skip if this isn't a valid point
        col = int((x - targetDSM.easting + 0.5*dx) / targetDSM.gsd)
        row = targetDSM.height - 1 - int((y-targetDSM.northing + 0.5*dy) / targetDSM.gsd)
        if col < 0:
            continue
        if row < 0:
            continue
        if col >= targetDSM.width - 1:
            continue
        if row >= targetDSM.height - 1:
            continue
        if targetDSM.data[row][col] == 0:
            continue
        targetZ = targetDSM.data[row][col] * targetDSM.scale + targetDSM.offset

        # Map the point into the reference
        # Skip if this isn't a valid point
        col = int((x - referenceDSM.easting + 0.5 * dx) / referenceDSM.gsd)
        row = referenceDSM.height - 1 - int((y - referenceDSM.northing + 0.5 * dy) / referenceDSM.gsd)
        if col < 0:
            continue
        if row < 0:
            continue
        if col >= referenceDSM.width - 1:
            continue
        if row >= referenceDSM.height - 1:
            continue
        if referenceDSM.data[row][col] == 0:
            continue
        referenceZ = referenceDSM.data[row][col] * referenceDSM.scale + referenceDSM.offset

        # Keep going until we have enough points
        difference = referenceZ - targetZ
        differences.append(difference)
        count += 1

    # Skip if not enough sampled points
    if count < numSamples:
        return False

    # Compute median Z offset and a robust estimate of the RMS difference
    rms = 0
    differences.sort()
    medianDZ = differences[int(count / 2)]  # TODO: double check if this could give non integer index
    for k in range(0, count):
        differences[k] = abs(differences[k] - medianDZ)

    differences.sort()
    rms = differences[int(count * 0.67)]

    # Compute the completeness
    good = 0
    for k in range(0, count):
        if differences[k] < 1.0:
            good += 1
    completeness = good / numSamples

    return True


# TODO: Debug and make EstimageRigidbody more efficient
def EstimateRigidBody(referenceDSM: OrthoImage, targetDSM: OrthoImage, maxt, bounds: AlignBounds, result: AlignResult):
    step = min(referenceDSM.gsd, targetDSM.gsd)
    numSamples = 10000
    maxSamples = numSamples * 10

    maxt = step * math.ceil(maxt / step)
    bins = int(maxt / step * 2) + 1
    
    # Initiate rmsArray
    rmsArray = np.empty((bins, bins))
    
    # Get random samples
    xlist = []
    ylist = []
    for i in range(0, maxSamples):
        random.seed(0)
        xlist.append(random.uniform(bounds.xmin, bounds.xmax))
        random.seed(0)
        ylist.append(random.uniform(bounds.ymin, bounds.ymax))

    # Start with brute force, but sample points to reduce timeline
    threshold = MAX_FLOAT
    bestDX = 0
    bestDY = 0
    bestDZ = 0
    besti = 0
    bestj = 0
    bestRMS = MAX_FLOAT
    medianDZ = 0
    bestRMS = threshold
    bestCompleteness = 0
    numSampled = 0
    for i in tqdm(range(0, bins)):
        dx = -maxt + i * step
        for j in range(0, bins):
            dy = -maxt + j * step
            rmsArray[i][j] = 0
            rms = 0
            completeness = 0
            ok =computeRMS(dx, dy, numSamples, maxSamples, xlist, ylist, referenceDSM,
                           targetDSM, medianDZ, rms, numSampled, completeness)
            if not ok:
                continue

            rmsArray[i][j] = rms
            if rms < bestRMS or (rms == bestRMS and (dx*dx + dy*dy < bestDX*bestDX+bestDY+bestDY)):
                bestCompleteness = completeness

                bestRMS = rms
                bestDX = dx
                bestDY = dy
                bestDZ = medianDZ
                besti = i
                bestj = j

    # Apply quadratic interpolation to localize the peak
    if besti > 0 and besti < bins - 1 and bestj > 0 and bestj < bins-1:
        dx = (rmsArray[besti + 1][bestj] - rmsArray[besti - 1][bestj]) / 2
        dy = (rmsArray[besti][bestj + 1] - rmsArray[besti][bestj - 1]) / 2
        dxx = (rmsArray[besti + 1][bestj] + rmsArray[besti - 1][bestj] - 2 * rmsArray[besti][bestj])
        dyy = (rmsArray[besti][bestj + 1] + rmsArray[besti][bestj - 1] - 2 * rmsArray[besti][bestj])
        dxy = (rmsArray[besti - 1][bestj + 1] - rmsArray[besti + 1][bestj - 1] - rmsArray[besti - 1][bestj + 1] + rmsArray[besti - 1][bestj - 1]) / 4
        det = dxx * dyy - dxy * dxy
        if det != 0:
            ix = besti - (dyy * dx - dxy * dy) / det
            iy = bestj - (dxx * dy - dxy * dx) / det
            bestDX = -maxt + ix * step
            bestDY = -maxt + iy * step

    # Deallocate RMS array
    del rmsArray

    # Update the result and return
    result.rms = bestRMS
    result.tx = -bestDX
    result.ty = -bestDY
    result.tz = bestDZ

    print(f"Percent less than 1m Z difference = {bestCompleteness * 100:6.2f}")
    print(f"X offset = {result.tx} m")
    print(f"Y offset = {result.ty} m")
    print(f"Z offset = {result.tz} m")
    print(f"Z RMS    = {result.rms} m")


def load_file(dsm: OrthoImage, inputFileName: Path, params: AlignParameters):
    ext = inputFileName.suffix
    if ext == ".tif":
        print(f"File Type = {ext}; loading as raster")
        if not dsm.read(inputFileName):
            print(f"Failed to read {str(inputFileName.absolute())}")
            return False
    else:
        # TODO: Point cloud to DSM, but probably unnecessary
        print("Cannot read point cloud...")
        return False
    # Fill small voids
    dsm.fillVoidsPyramid(True, 2)
    print("Filtering data...")
    # Remove points along edges which are difficult to match
    dsm.edgeFilter(int(params.maxdz / dsm.scale))
    return True


def AlignTarget2Reference(referenceFilename: Path, targetFilename: Path, params: AlignParameters):
    print(f"Reading reference file {str(referenceFilename.absolute())}")
    referenceDSM = OrthoImage(TYPE=np.ushort)
    if not load_file(referenceDSM, referenceFilename, params):
        return False
    if params.gsd != referenceDSM.gsd:
        print(f"Changing gsd to {referenceDSM.gsd} to match reference DSM")
        params.gsd = referenceDSM.gsd

    print(f"Reading target file: {str(targetFilename.absolute())}")
    targetDSM = OrthoImage(TYPE=np.ushort)
    if not load_file(targetDSM, targetFilename, params):
        return False

    # Get overlapping bounds
    bounds = AlignBounds()
    bounds.xmin = max(referenceDSM.easting, targetDSM.easting)
    bounds.ymin = max(referenceDSM.northing, targetDSM.northing)
    bounds.xmax = min(referenceDSM.easting + (referenceDSM.width * referenceDSM.gsd),
                      targetDSM.easting + (targetDSM.width * targetDSM.gsd))
    bounds.ymax = min(referenceDSM.northing + (referenceDSM.height * referenceDSM.gsd),
                      targetDSM.northing + (targetDSM.height * targetDSM.gsd))
    bounds.width = bounds.xmax - bounds.xmin
    bounds.height = bounds.ymax - bounds.ymin
    overlap_km = bounds.width / 1000 * bounds.height / 1000
    print(f"Overlap = {int(bounds.width)} m x {int(bounds.height)} m = {overlap_km} km")
    if overlap_km == 0:
        return False

    # Estimate rigid body transform to align target points to reference
    result = AlignResult()
    print("Estimating rigid body transformation.")
    EstimateRigidBody(referenceDSM, targetDSM, params.maxt, bounds, result)

    # Write offsets to text file
    print("Writing offsets text file.s")
    f1= open(Path(targetFilename.parent, targetFilename.stem + "_offsets.txt"), 'a')
    f1.write("X Offset  Y Offset  Z Offset  Z RMS\n")
    f1.write("%08.3f  %08.3f  %08.3f  %08.3f\n" % (result.tx, result.ty, result.tz, result.rms))
    f1.close()

    # Write aligned TIF file
    print("Writing aligned TIF file.")
    outFileName = Path(targetFilename.parent, targetFilename.stem + "_aligned.tif")
    targetDSM.offset += result.tz
    targetDSM.easting += result.tx
    targetDSM.northing += result.ty
    targetDSM.write(outFileName, True)

    # Write BPF File
    # TODO: Write BPF File

    return True


def main():
    referenceDSMfilename = Path(r"C:\Users\wangss1\Documents\Data\CORE3D_Phase1B_Extension\Testing_Data\GroundTruth\A-1\A-1_DSM.tif")
    targetDSMfilename = Path(r"C:\Users\wangss1\Documents\Data\CORE3D_Phase1B_Extension\Testing_Data\VRICON\A-1\a-1_dsm.tif")
    params = AlignParameters()
    AlignTarget2Reference(referenceDSMfilename, targetDSMfilename, params)
if __name__ == "__main__":
    main()