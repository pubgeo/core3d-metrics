import gdal
import numpy as np
import os


def imageLoad(filename):
    im = gdal.Open(filename, gdal.GA_ReadOnly)
    band = im.GetRasterBand(1)
    img = band.ReadAsArray(0, 0, im.RasterXSize, im.RasterYSize)
    transform = im.GetGeoTransform()
    return img, transform


def getNoDataValue(filename):
    im = gdal.Open(filename, gdal.GA_ReadOnly)
    band = im.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    return nodata


def imageWarp(file_from: str, file_to: str, offset=None, interp_method: int = gdal.gdalconst.GRA_Bilinear, noDataValue=None):
    image_from = gdal.Open(file_from, gdal.GA_ReadOnly)
    image_to = gdal.Open(file_to, gdal.GA_ReadOnly)

    # Apply registration offset
    if offset is not None:
        # Move input to memory to apply registration offset
        mem_drv0 = gdal.GetDriverByName('MEM')
        image_tmp = mem_drv0.Create('', image_from.RasterXSize,
                                    image_from.RasterYSize, 1, gdal.GDT_Float32)
        image_tmp.SetGeoTransform(image_from.GetGeoTransform())
        image_tmp.SetProjection(image_from.GetProjection())
        image_tmp.GetRasterBand(1).WriteArray(
            image_from.ReadAsArray(0, 0, image_from.RasterXSize,
                                   image_from.RasterYSize))
        NDV = image_from.GetRasterBand(1).GetNoDataValue()
        if NDV is not None:
            image_tmp.GetRasterBand(1).SetNoDataValue(NDV)

        offset = np.asarray(offset)
        transform = image_from.GetGeoTransform()
        transform = np.asarray(transform)
        transform[0] += offset[0]
        transform[3] += offset[1]
        image_tmp.SetGeoTransform(transform)
    else:
        image_tmp = image_from

    # Create output image
    mem_drv = gdal.GetDriverByName('MEM')
    destination = mem_drv.Create('', image_to.RasterXSize, image_to.RasterYSize, 1,
                          gdal.GDT_Float32)

    destination.SetProjection(image_to.GetProjection())
    destination.SetGeoTransform(image_to.GetGeoTransform())

    if noDataValue is not None:
        band = destination.GetRasterBand(1);
        band.SetNoDataValue(noDataValue)
        band.Fill(noDataValue)
    
    gdal.ReprojectImage(image_tmp, destination, image_from.GetProjection(),
                        image_to.GetProjection(), interp_method)

    image_out = destination.GetRasterBand(1).ReadAsArray(0, 0, destination.RasterXSize, destination.RasterYSize)

    return image_out


def arrayToGeotiff(image_array, out_file_name, reference_file_name, NODATA_VALUE):
    """ Used to save rasterized dsm of point cloud """
    reference_image = gdal.Open(reference_file_name, gdal.GA_ReadOnly)
    transform = reference_image.GetGeoTransform()
    projection = reference_image.GetProjection()

    driver = gdal.GetDriverByName('GTiff')
    out_image = driver.Create(out_file_name + '.tif', image_array.shape[1],
                              image_array.shape[0], 1, gdal.GDT_Float32)
    if out_image is None:
        print('Could not create output GeoTIFF')

    out_image.SetGeoTransform(transform)
    out_image.SetProjection(projection)

    out_band = out_image.GetRasterBand(1)
    out_band.SetNoDataValue(NODATA_VALUE)
    out_band.WriteArray(image_array, 0, 0)
    out_band.FlushCache()
    out_image.FlushCache()
    # Ignore pep warning here, aids in memory management performance
    out_image = None

    return


# Load LAS file and generate max DSM in memory
def lasToRaster(las_filename, transform, shape_out, NODATA):
    # Load LAS file
    test_las = File(las_filename, mode='r')

    x = test_las.x
    y = test_las.y
    z = test_las.z

    # Project to output image space
    # TODO: call map2pix
    map_to_pix = gdal.InvGeoTransform(transform)
    x0 = np.round(map_to_pix[0] + x * map_to_pix[1] + y * map_to_pix[2])
    y0 = np.round(map_to_pix[3] + x * map_to_pix[4] + y * map_to_pix[5])

    x0 = x0.astype(int)
    y0 = y0.astype(int)

    # Generate MAX value DSM
    raster = np.zeros(shape_out, np.float32) + NODATA
    for ii in range(0, x0.size):
        if (x0[ii] >= 0) & (x0[ii] < raster.shape[1]) & (y0[ii] >= 0) & (
                y0[ii] < raster.shape[0]):
            if z[ii] > raster[y0[ii], x0[ii]]:
                raster[y0[ii], x0[ii]] = z[ii]

    return raster


# refMat is a GDAL GeoTransform format
def map2pix(reference_matrix, points_list):
    x_origin = reference_matrix[0]
    y_origin = reference_matrix[3]
    pixel_width = reference_matrix[1]
    pixel_height = -reference_matrix[5]

    xy = np.zeros(shape=(len(points_list), 2))

    xy[:, 0] = (np.round((points_list[:, 0] - x_origin) / pixel_width))
    xy[:, 1] = (np.round((y_origin - points_list[:, 1]) / pixel_height))

    return xy
