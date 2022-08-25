import os
import gdal, osr
import numpy as np


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


def getMetadata(inputinfo):

    # dataset input
    if isinstance(inputinfo,gdal.Dataset):
        dataset = inputinfo
        FLAG_CLOSE = False

    # file input
    elif isinstance(inputinfo,str):
        filename = inputinfo
        if not os.path.isfile(filename):
            raise IOError('Cannot locate file <{}>'.format(filename))

        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        FLAG_CLOSE = True

    # unrecognized input
    else:
        raise IOError('Unrecognized getMetadata input')

    # read metadata
    meta = {
        'RasterXSize':  dataset.RasterXSize,
        'RasterYSize':  dataset.RasterYSize,
        'RasterCount':  dataset.RasterCount,
        'Projection':   dataset.GetProjection(),
        'GeoTransform': list(dataset.GetGeoTransform()),
        'BitDepth': dataset.GetRasterBand(1).DataType,
        'EPSG': osr.SpatialReference(wkt=dataset.GetProjection()).GetAttrValue('AUTHORITY',1)
    }

    # cleanuo
    if FLAG_CLOSE: dataset = None
    return meta

def imageWarpRGB(file_src: str, file_dst: str, offset=None, interp_method: int = gdal.gdalconst.GRA_Bilinear, noDataValue=None):

    # verbose display
    print('Loading <{}>'.format(file_src))

    # destination metadata
    meta_dst = getMetadata(file_dst)

    # GDAL memory driver
    mem_drv = gdal.GetDriverByName('MEM')

    # copy source to memory
    tmp = gdal.Open(file_src, gdal.GA_ReadOnly)
    dataset_src = mem_drv.CreateCopy('',tmp)
    tmp = None

    # source metadata
    meta_src = getMetadata(dataset_src)

    # Apply registration offset
    if offset is not None:

        # offset error: offset is defined in destination projection space,
        # and cannot be applied if source and destination projections differ
        if meta_src['Projection'] != meta_dst['Projection']:
            print('IMAGE PROJECTION\n{}'.format(meta_src['Projection']))
            print('OFFSET PROJECTION\n{}'.format(meta_dst['Projection']))
            raise ValueError('Image/Offset projection mismatch')

        transform = meta_src['GeoTransform']
        transform[0] += offset[0]
        transform[3] += offset[1]
        dataset_src.SetGeoTransform(transform)


    # no reprojection necessary
    if meta_src == meta_dst:
        print('  No reprojection')
        dataset_dst = dataset_src

    # reprojection
    else:
        keys = [k for k in meta_dst if meta_dst.get(k) != meta_src.get(k)]
        print('  REPROJECTION (adjusting {})'.format(', '.join(keys)))

        # file, xsz, ysz, nbands, dtype
        dataset_dst = mem_drv.Create('', meta_dst['RasterXSize'], meta_dst['RasterYSize'],
            meta_src['RasterCount'], gdal.GDT_Float32)

        dataset_dst.SetProjection(meta_dst['Projection'])
        dataset_dst.SetGeoTransform(meta_dst['GeoTransform'])

        # input, output, inputproj, outputproj, interp
        gdal.ReprojectImage(dataset_src, dataset_dst, meta_src['Projection'],
             meta_dst['Projection'], interp_method)

    # read & return image data
    r = dataset_dst.GetRasterBand(1).ReadAsArray()
    g = dataset_dst.GetRasterBand(2).ReadAsArray()
    b = dataset_dst.GetRasterBand(3).ReadAsArray()
    img = np.dstack((r,g,b))
    img = np.uint8(img)
    return img

def imageWarp(file_src: str, file_dst: str, offset=None, interp_method: int = gdal.gdalconst.GRA_Bilinear, noDataValue=None):

    # verbose display
    print('Loading <{}>'.format(file_src))

    # destination metadata
    meta_dst = getMetadata(file_dst)

    # GDAL memory driver
    mem_drv = gdal.GetDriverByName('MEM')

    # copy source to memory
    tmp = gdal.Open(file_src, gdal.GA_ReadOnly)
    dataset_src = mem_drv.CreateCopy('',tmp)   
    tmp = None

    # change no data value to new "noDataValue" input if necessary,
    # making sure to adjust the underlying pixel values
    band = dataset_src.GetRasterBand(1)
    NDV = band.GetNoDataValue()

    if noDataValue is not None and noDataValue != NDV:
        if NDV is not None:            
            img = band.ReadAsArray()
            img[img==NDV] = noDataValue
            band.WriteArray(img)
        band.SetNoDataValue(noDataValue)        
        NDV = noDataValue

    # source metadata
    meta_src = getMetadata(dataset_src)

    # Reproject if dst and source do not have matching projections. Reproject to dst
    if meta_src['Projection'] != meta_dst['Projection'] or meta_src['RasterXSize'] != meta_dst["RasterXSize"]\
            or meta_src["RasterYSize"] != meta_dst["RasterYSize"]:
        print('IMAGE PROJECTION\n{}'.format(meta_src['Projection']))
        print('OFFSET PROJECTION\n{}'.format(meta_dst['Projection']))
        # raise ValueError('Image/Offset projection mismatch')

        # Reproject
        keys = [k for k in meta_dst if meta_dst.get(k) != meta_src.get(k)]
        print('  REPROJECTION (adjusting {})'.format(', '.join(keys)))

        # file, xsz, ysz, nbands, dtype
        dataset_dst = mem_drv.Create('', meta_dst['RasterXSize'], meta_dst['RasterYSize'],
                                     meta_src['RasterCount'], gdal.GDT_Float32)

        dataset_dst.SetProjection(meta_dst['Projection'])
        dataset_dst.SetGeoTransform(meta_dst['GeoTransform'])

        if NDV is not None:
            band = dataset_dst.GetRasterBand(1)
            band.SetNoDataValue(NDV)
            band.Fill(NDV)

        # input, output, inputproj, outputproj, interp
        gdal.ReprojectImage(dataset_src, dataset_dst, meta_src['Projection'],
                            meta_dst['Projection'], interp_method)
    else:
        dataset_dst = dataset_src

    # Apply registration offset
    if offset is not None:
        # offset error: offset is defined in destination projection space,
        # and cannot be applied if source and destination projections differ
        transform = meta_src['GeoTransform']
        transform[0] += offset[0]
        transform[3] += offset[1]
        dataset_src.SetGeoTransform(transform)

    # read & return image data
    img = dataset_dst.GetRasterBand(1).ReadAsArray()    
    return img


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

def arrayToGeotiffRGB(image_array, out_file_name, reference_file_name, NODATA_VALUE):
    """ Used to save rasterized dsm of point cloud """
    reference_image = gdal.Open(reference_file_name, gdal.GA_ReadOnly)
    transform = reference_image.GetGeoTransform()
    projection = reference_image.GetProjection()

    bands = image_array.shape[2]
    driver = gdal.GetDriverByName('GTiff')
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']

    out_image = driver.Create(out_file_name + '.tif', image_array.shape[1],
                              image_array.shape[0], bands, gdal.GDT_Byte, options=options)
    if out_image is None:
        print('Could not create output GeoTIFF')

    out_image.SetGeoTransform(transform)
    out_image.SetProjection(projection)

    for band in range(bands):
        out_image.GetRasterBand(band+1).WriteArray(image_array[:, :, band])

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
