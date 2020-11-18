import gdal, ogr, gdalconst, osr
from pathlib import Path
import re
import sys
import numpy as np

MAX_INT = np.iinfo(int).max
MAX_FLOAT = np.finfo(float).max


def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None: 'None',
            gdal.CE_Debug: 'Debug',
            gdal.CE_Warning: 'Warning',
            gdal.CE_Failure: 'Failure',
            gdal.CE_Fatal: 'Fatal'
    }
    err_msg = err_msg.replace('\n', ' ')
    err_class = errtype.get(err_class, 'None')
    print('Error Number: %s' % err_num)
    print('Error Type: %s' % err_class)
    print('Error Message: %s' % err_msg)


def get_max_value_of_datatype(datatype):
    if datatype == 'Int8' or datatype == np.int8:
        return 127
    elif datatype == 'Uint8' or datatype == np.uint8:
        return 255
    elif datatype == 'UInt16' or datatype == np.ushort:
        return 65535
    elif datatype == 'Int16' or datatype == np.short:
        return 32767
    elif datatype == 'Float32' or datatype == np.float32:
        return 3.402823466 * 10 ** 38
    else:
        return None


def get_gdal_type_from_type(image_type):
    if image_type == np.uint16 or image_type == np.ushort:
        return 2
    if image_type == np.uint8:
        return 1
    if image_type == np.float64:
        return 7
    if image_type == np.float32:
        return 6
    if image_type == np.uint32:
        return 4


class OrthoImage:

    # Default Constructor
    def __init__(self, TYPE=None, OrthoImage=None, easting=None, northing=None, zone=None, gsd=None):
        if OrthoImage is not None:
            self.easting = OrthoImage.easting
            self.northing = OrthoImage.northing
            self.zone = OrthoImage.zone
            self.gsd = OrthoImage.gsd
        self.TYPE = TYPE
        self.easting = easting
        self.northing = northing
        self.zone = zone
        self.gsd = gsd
        self.width = None
        self.height = None
        self.bands = None
        self.offset = None
        self.scale = None
        self.data = None
        self.projection = None

    # Read any GDAL-supported image
    def read(self, filename: Path):
        # Open the image
        poDataset = gdal.Open(str(filename.absolute()), gdalconst.GA_ReadOnly)
        # Get geospatial metadata
        print(f"Driver: {poDataset.GetDriver().GetDescription()} {poDataset.GetDriver().LongName}")
        self.width = poDataset.RasterXSize
        self.height = poDataset.RasterYSize
        self.bands = poDataset.RasterCount
        print(f"Width = {self.width}\nHeight = {self.height}\nBands = {self.bands}")
        projection = poDataset.GetProjection()
        self.projection = projection
        print(f"Projection is {projection}")
        adfGeoTransform = poDataset.GetGeoTransform()
        print(f"GeoTransform = {adfGeoTransform[0]}, {adfGeoTransform[1]}, {adfGeoTransform[2]}, "
              f"{adfGeoTransform[3]}, {adfGeoTransform[4]}, {adfGeoTransform[5]}")
        xscale = adfGeoTransform[1]
        yscale = -adfGeoTransform[5]
        self.easting = adfGeoTransform[0] - adfGeoTransform[2] * xscale
        self.northing = adfGeoTransform[3] - self.height * yscale
        myOGRS = poDataset.GetProjectionRef()
        srs = osr.SpatialReference(wkt=myOGRS)
        if srs.IsProjected:
            projection_string = srs.GetAttrValue('projcs')
            # regex to find numbers followed by a single letter in projcs string
            pattern = re.compile('\d+[a-zA-Z]{1}')
            result = re.search(pattern, projection_string)
            result_str = result.group()
            zone_number = int(result_str[:-1])
            zone_hemisphere = result_str[-1]
            if zone_hemisphere is "S":
                self.zone = zone_number * -1
            elif zone_hemisphere is "N":
                self.zone = zone_number
        else:
            print("Image is not projected. Returning None...")

        self.gsd = (xscale+yscale) / 2
        print(f"UTM Easting = {self.easting}\nUTM Northing = {self.northing}\n"
              f"UTM Zone = {self.zone}\nGSD = {self.gsd}")

        # Get band information
        poBand = poDataset.GetRasterBand(1)
        if poBand is None:
            print("Error opening first band...")
        BandDataType = gdal.GetDataTypeName(poBand.DataType)

        noData = poBand.GetNoDataValue()
        # TODO: Check for floating point precision
        if noData is None:
            if self.TYPE is float:
                # Set noData only for floating point images
                noData = float(-10000)
            else:
                noData = 0

        # Get scale and offset values
        if self.TYPE is float:
            # Do not scale if floating point values
            self.scale = 1
            self.offset = 0
        else:
            adfMinMax = [0, 0]
            first_pass = True
            minVal = None
            maxVal = None
            for i in range(0, self.bands):
                poBand = poDataset.GetRasterBand(i+1)
                if poBand is None:
                    print(f"Error opening band {i+1}")

                adfMinMax[0] = None #poBand.GetMinimum()
                adfMinMax[1] = None #poBand.GetMaximum()
                if not adfMinMax[0] or not adfMinMax[1]:
                    min, max = poBand.ComputeRasterMinMax(True)
                    adfMinMax[0] = min
                    adfMinMax[1] = max
                if first_pass:
                    minVal = adfMinMax[0]
                    maxVal = adfMinMax[1]
                    first_pass = False
                else:
                    if minVal > adfMinMax[0]:
                        minVal = float(adfMinMax[0])
                    if maxVal < adfMinMax[1]:
                        maxVal = float(adfMinMax[1])
            # Reserve zero fo noData value
            minVal -= 1
            maxVal += 1
            # TODO: Remove manual BandDataType, acquired above correctly.
            # BandDataType = 'UInt16'
            # maxImageVal  = float(pow(2.0, int((np.iinfo(self.TYPE).max) * 8)) - 1)
            maxImageVal = get_max_value_of_datatype(self.TYPE)
            self.offset = minVal
            self.scale = (maxVal - minVal) / maxImageVal
        print(f"Offset = {self.offset}")
        print(f"Scale = {self.scale}")

        # Read the image, one band at a time
        offset_function = lambda x: (x-self.offset)/self.scale
        vfunc = np.vectorize(offset_function)
        for ib in range(0, self.bands):
            # Read the next row
            poBand = poDataset.GetRasterBand(ib+1)
            raster = poBand.ReadAsArray()
            shifted_array = vfunc(raster)
            self.data = shifted_array
        return True

    def fillVoidsPyramid(self, noSmoothing: bool, maxLevel=MAX_INT):
        # Check for voids
        count = np.count_nonzero(self.data == 0)
        if count == 0:
            return

        # Create image pyramid
        pyramid = []
        pyramid.append(self)
        level = 0
        while count > 0 and level < maxLevel:
            # Create next level
            nextWidth = int(pyramid[level].width / 2)
            nextHeight = int(pyramid[level].height / 2)

            newImagePtr = OrthoImage(self.TYPE)
            newImagePtr.data = np.zeros((nextWidth, nextHeight)).astype(self.TYPE)

            # Fill in non-void values from level below building up the pyramid with a simple running average
            for i in range(0, nextHeight):
                for j in range(0, nextWidth):
                    j2 = min(max(0, j * 2 + 1), pyramid[level].height - 1)
                    i2 = min(max(0, i * 2 + 1), pyramid[level].width - 1)

                    # Average neighboring pixels from below
                    z = 0
                    ct = 0
                    neighbors = []
                    for jj in range(max(0, j2 - 1), min(j2+1, pyramid[level].height - 1)):
                        for ii in range(max(0, i2 - 1), min(i2 + 1, pyramid[level].width -1)):
                            if pyramid[level].data[jj][ii] != 0:
                                z += pyramid[level].data[jj][ii]
                                ct += 1
                    if ct != 0:
                        z = z / ct
                        newImagePtr.data[j][i] = self.TYPE(z)
            pyramid.append(newImagePtr)
            level += 1
            count = np.count_nonzero(pyramid[level] == 0)

        # Void fill down the pyramid
        for k in range(level-1, 0, -1):
            ref = OrthoImage(OrthoImage=pyramid[k])
            for j in range(0, pyramid[k].height):
                for i in range(0, pyramid[k].width):
                    if pyramid[k].data[j][i] == 0:
                        j2 = min(max(0, j / 2), pyramid[k+1].height - 1)
                        i2 = min(max(0, i / 2), pyramid[k+1].width - 1)

                        if noSmoothing:
                            # Just use the closest pixel from above
                            pyramid[k].data[j][i] = pyramid[k+1].data[j2][i2]
                        else:
                            # Averate neighboring pixels from around and above
                            wts = 0
                            ttl = 0
                            # TODO: wtf...
                            if j > 0:
                                for j3 in range(j-1, j+1):
                                    if i > 0:
                                        for i3 in range(i-1, i+1):
                                            z = 0
                                            if j3 >= 0 and i3 >= 0:
                                                if j3 < pyramid[k].height and i3 < pyramid[k].width:
                                                    z = ref.data[j3][i3]
                                                if not z and j3/2 < pyramid[k+1].height and i3/2 < pyramid[k+1].width:
                                                    z = pyramid[k+1].data[j3/2][i3/2]
                                                if z:
                                                    w = 1 + 1 * (i3 == i or j3 == j)
                                                    ttl += w*z
                                                    wts += w
                                    else:
                                        for i3 in range(0, i+1):
                                            z = 0
                                            if j3 >= 0 and i3 >= 0:
                                                if j3 < pyramid[k].height and i3 < pyramid[k].width:
                                                    z = ref.data[j3][i3]
                                                if not z and j3 / 2 < pyramid[k + 1].height and i3 / 2 < pyramid[
                                                    k + 1].width:
                                                    z = pyramid[k + 1].data[j3 / 2][i3 / 2]
                                                if z:
                                                    w = 1 + 1 * (i3 == i or j3 == j)
                                                    ttl += w * z
                                                    wts += w
                            else:
                                for j3 in range(0, j+1):
                                    if i > 0:
                                        for i3 in range(i-1, i+1):
                                            z = 0
                                            if j3 >= 0 and i3 >= 0:
                                                if j3 < pyramid[k].height and i3 < pyramid[k].width:
                                                    z = ref.data[j3][i3]
                                                if not z and j3 / 2 < pyramid[k + 1].height and i3 / 2 < pyramid[
                                                    k + 1].width:
                                                    z = pyramid[k + 1].data[j3 / 2][i3 / 2]
                                                if z:
                                                    w = 1 + 1 * (i3 == i or j3 == j)
                                                    ttl += w * z
                                                    wts += w
                                    else:
                                        for i3 in range(0, i+1):
                                            z = 0
                                            if j3 >= 0 and i3 >= 0:
                                                if j3 < pyramid[k].height and i3 < pyramid[k].width:
                                                    z = ref.data[j3][i3]
                                                if not z and j3 / 2 < pyramid[k + 1].height and i3 / 2 < pyramid[
                                                    k + 1].width:
                                                    z = pyramid[k + 1].data[j3 / 2][i3 / 2]
                                                if z:
                                                    w = 1 + 1 * (i3 == i or j3 == j)
                                                    ttl += w * z
                                                    wts += w
                            if wts:
                                pyramid[k].data[j][i] = ttl / wts

        # Deallocate memory for all but the input dsm
        for i in range(1, level):
            del pyramid[i]

    def edgeFilter(self, dzScaled):
        dzScaled = self.TYPE(dzScaled)
        # TODO: Apply filter to the image
        return

    def write(self, filename: Path, convertToFloat=False, egm96=False):
        if convertToFloat:
            target_ds = gdal.GetDriverByName('GTiff').Create(
                str(filename.absolute()), self.width, self.height, 1, gdal.GDT_Float64)
        else:
            target_ds = gdal.GetDriverByName('GTiff').Create(
                str(filename.absolute()), self.width, self.height, 1, get_gdal_type_from_type(self.TYPE))

        adfGeoTransform = [self.easting, self.gsd, 0, self.northing + self.height * self.gsd, 0, -1 * self.gsd]
        target_ds.SetGeoTransform(adfGeoTransform)
        target_ds.SetProjection(self.projection)
        target_ds.GetRasterBand(1).WriteArray(self.data)

        band = target_ds.GetRasterBand(1)
        no_data_value = 0
        band.SetNoDataValue(no_data_value)
        band.FlushCache()
        target_ds = None
        return True


def main():
    filename = Path(r"C:\Users\wangss1\Documents\Data\CORE3D_Phase1B_Extension\Testing_Data\GroundTruth\A-1\A-1_DSM.tif")
    test = OrthoImage(TYPE=np.ushort)
    test.read(filename)


if __name__ == "__main__":
    main()
