import json
from pathlib import Path
from xml.etree import ElementTree

from numpy import uint16
import numpy as np
from osgeo import ogr, gdal, osr
import os

_print_once = {}


# Helper function to keep logs minimal
def print_once(string="", dump_override=False):
    if dump_override:
        for key, value in _print_once.items():
            if value > 1:
                print("{}: {} times".format(key, value))
            else:
                print("{}".format(key))
    else:
        if string in _print_once:
            _print_once[string] += 1
        else:
            _print_once[string] = 1


def parse_osm(in_path):
    if not in_path.exists() or not in_path.is_file():
        raise FileNotFoundError("OSM path provided is not valid", in_path.absolute())
    tree = ElementTree.parse(in_path)
    root = tree.getroot()

    nodes = {}
    ways = []
    relations = []

    for child in root:
        if child.tag == 'node':
            lat = child.attrib['lat']
            lon = child.attrib['lon']
            nodes[child.attrib['id']] = (lon, lat)
        elif child.tag == 'way':
            way_points = []
            way_features = {}
            for way_child in child:
                if way_child.tag == 'nd':
                    ref = way_child.attrib['ref']
                    way_points.append(ref)
                elif way_child.tag == 'tag':
                    way_features[way_child.attrib['k']] = way_child.attrib['v']
                else:
                    print_once("Unknown way child node: {}".format(way_child.tag))
            # If BuildingID/ID already tagged, need to truncate to uint16 (then back to int for json purposes)
            if "BuildingID" in way_features.keys():
                way_features["BuildingID"] = int(uint16(way_features["BuildingID"]))
            else:
                way_features["BuildingID"] = int(uint16(child.attrib['id']))
            ways.append((way_points, way_features))
        elif child.tag == 'relation':
            # TODO: handle relations, Priority LOW
            relations.append(child)
            print_once("Unhandled relation")
        else:
            print_once("Unknown tag: {}".format(child.tag))

    return ways, nodes


def osm_to_geojson(ways, nodes):
    # TODO: this could be beefed up, but... it's about as accurate as the solutions proposed by osm wiki. Priority MED
    geojson_object = {
        "type": "FeatureCollection",
        "Properties": "Converted",
        "features": []
    }

    for way in ways:
        way_points, features = way
        coordinates = []
        if 'building' not in features:
            continue
        for node in way_points:
            if node not in nodes:
                print_once("Missing node reference: {}".format(node))
                continue
            lon, lat = nodes[node]
            lat, lon, _ = ll2utm(lat, lon)
            coordinates.append([float(lon), float(lat)])
        if len(coordinates) <= 0:
            raise RuntimeError("Invalid geospatial object with 0 points")
        elif len(coordinates) == 1:
            # TODO: implement geojson point: PRIORITY LOW
            raise NotImplemented("Points")
        else:
            geometry_type = "LineString"
            if coordinates[0] == coordinates[-1] and len(coordinates) >= 4:
                geometry_type = "Polygon"
            else:
                # TODO: accept linestrings, but i think there's currently a minor error
                continue

            # TODO: add nested polygons to coordinates for holes / mutli-polygons: PRIORITY MED
            # Replace [coordinates] with [coordinates, holes]
            geojson_feature = \
                {
                    "type": "Feature",
                    "properties": features,
                    "geometry": {
                        "type": geometry_type,
                        "coordinates": [coordinates]
                    }
                }
        geojson_object['features'].append(geojson_feature)
    return geojson_object


def osm_to_ogr(osm_path):
    gdal.PushErrorHandler(gdal_error_handler)
    # First convert to geojson
    geojson = osm_to_geojson(*parse_osm(osm_path))
    # Read geojson as ogr geometry
    ogr_source = ogr.CreateGeometryFromJson(json.dumps(geojson))
    return ogr_source


def osm_file_to_np_array(osm_path):
    ogr_source = osm_to_ogr(osm_path)
    # todo: try this without a made up source tif
    # otherwise, generate a temporary tiff with 2048^2 0.5m resolution
    return create_raster_from_ogr_source_and_geotiff(ogr_source)


def osm_file_to_tiff(osm_path, tiff_path=None, x_size=2048, y_size=2048, res=0.5):
    ogr_source = osm_to_ogr(osm_path)
    if tiff_path is None:
        tiff_path = osm_path.with_suffix(".tif")


def create_raster_from_geojson(geojson_path, source_raster_path=None, no_data_value=0, x_res=None, y_res=None):
    """
    Converts a geojson vector product into a raster stored as a numpy array
    :param geojson_path: System path to geojson to be converted
    :param source_raster_path: Geotiff path to copy geo-reference metadata from
    :param no_data_value: No data value to label pixels with no data. Default= 0
    :return: numpy array
    """
    source = ogr.Open(geojson_path)
    if source is not None:
        if source_raster_path is None:
            return create_raster_from_ogr_source(source, no_data_value=2, pixel_size=0.5, x_res=x_res, y_res=y_res)
        else:
            return create_raster_from_ogr_source_and_geotiff(source, source_raster_path, no_data_value)
    else:
        print("Cannot find source file: " + geojson_path)
        return None


def create_raster_from_ogr_source_and_geotiff(source, source_raster_path=None, no_data_value=0, x_size=2048, y_size=2048, res=0.5):
    """
    Converts a geojson vector product into a raster stored as a numpy array
    :param source: OGR Geometry
    :param source_raster_path: Geotiff path to copy geo-reference metadata from
    :param no_data_value: No data value to label pixels with no data. Default= 0
    :param x_size: The width of the output raster
    :param y_size: The width of the output raster
    :return: numpy array
    """
    source_layer = source.GetLayer()
    geo_transform = None
    projection = None
    if source_raster_path is not None:
        source_raster = gdal.Open(source_raster_path)  # needs to be a geotiff
        x_size = source_raster.RasterXSize
        y_size = source_raster.RasterYSize
        geo_transform = source_raster.GetGeoTransform()
        projection = source_raster.GetProjection()

    # Create the destination data source
    target = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_UInt16)

    # todo: this if statement may be unnecessary (gdal may allow setting things to None gracefully)
    if source_raster_path is not None:
        target.SetGeoTransform(geo_transform)
        target.SetProjection(projection)

    band = target.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    gdal.RasterizeLayer(target, [1], source_layer, options=["ATTRIBUTE=BuildingID"])
    raster = band.ReadAsArray()
    return raster


# example GDAL error handler function
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


def ll2utm(lat, lon, datum='wgs84'):
    """
    :param lat: latitude in degrees (float 64)
    :param lon: longitude in degrees (float64)
    :return: utm-x coordinate, utm-y coordinate, zone number (positive for North, negative for South)
    """
    lat = float(lat)
    lon = float(lon)
    datums = {
        'wgs84': [6378137.0, 298.257223563],
        'nad83': [6378137.0, 298.257222101],
        'grs80': [6378137.0, 298.257222101],
        'nad27': [6378206.4, 294.978698214],
        'int24': [6378388.0, 297.000000000],
        'clk66': [6378206.4, 294.978698214]
    }

    # Constants
    D0 = 180/np.pi  # conversion rad to deg
    K0 = 0.9996  # UTM scale factor
    X0 = 500000  # UTM false EAST (m)

    # defaults
    zone = None

    if datum in datums.keys():
        A1 = datums['wgs84'][0]
        F1 = datums['wgs84'][1]

    p1 = lat/D0
    l1 = lon/D0

    # UTM zone automatic setting
    if zone is None:
        F0 = np.round((l1*D0 + 183)/6)
    else:
        F0 = zone

    B1 = A1*(1-(1/F1))
    E1 = np.sqrt((A1 * A1 - B1 * B1)/(A1 * A1))
    P0 = 0/D0
    L0 = (6*F0 - 183)/D0  # UTM origin longitude (rad)
    Y0 = 10000000*(p1 < 0)  # UTM false northern (m)
    N = K0*A1

    C = calculate_projection_coefficients(E1, 0)
    B = C[0]*P0 + C[1]*np.sin(2*P0) + C[2]*np.sin(4*P0) + C[3]*np.sin(6*P0) + C[4]*np.sin(8*P0)

    YS = Y0 - N * B
    C = calculate_projection_coefficients(E1, 2)
    L = np.log(np.tan(np.pi / 4 + p1 / 2) * (((1 - E1 * np.sin(p1)) / (1 + E1 * np.sin(p1))) ** (E1 / 2)))
    z = (np.arctan(np.sinh(L) / np.cos(l1 - L0))) + (1j * np.log(np.tan(np.pi / 4 + np.arcsin(np.sin(l1 - L0) / np.cosh(L)) / 2)))  # complex number
    Z = N * C[0] * z + N * (C[1] * np.sin(2 * z) + C[2] * np.sin(4 * z) + C[3] * np.sin(6 * z) + C[4] * np.sin(8 * z))
    xs = np.imag(Z) + X0
    ys = np.real(Z) + YS

    f = F0 * np.sign(lat)
    fu = np.unique(f)
    if np.size(fu) == 1 and np.isscalar(fu[0]):
        f = fu[0]
    x = xs[0]
    y = ys[0]

    return x, y, f


def calculate_projection_coefficients(e, m):
    """
    COEF Projection coefficients
    calculate_projection_coefficients(e,m) returns a vector of 5 coefficients
    :param e: first ellipsoid excentricity
    :param m: m=0 for tranverse mercator, m=1 for transverse mercator reverse, m=2 for merdian arc
    :return: c = numpy array of length 5, projection coefficients
    """
    if m == 0:
        c0 = np.array([[-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
                       [-105 / 4096, 0, -45 / 1024, 0, -3 / 32, 0, -3 / 8, 0, 0],
                       [525 / 16384, 0, 45 / 1024, 0, 15 / 256, 0, 0, 0, 0],
                       [-175 / 12288, 0, -35 / 3072, 0, 0, 0, 0, 0, 0],
                       [315 / 131072, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif m == 1:
        c0 = np.array([[-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
                       [1 / 61440, 0, 7 / 2048, 0, 1 / 48, 0, 1 / 8, 0, 0],
                       [559 / 368640, 0, 3 / 1280, 0, 1 / 768, 0, 0, 0, 0],
                       [283 / 430080, 0, 17 / 30720, 0, 0, 0, 0, 0, 0],
                       [4397 / 41287680, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif m == 2:
        c0 = np.array([[-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
                       [-901 / 184320, 0, -9 / 1024, 0, -1 / 96, 0, 1 / 8, 0, 0],
                       [-311 / 737280, 0, 17 / 5120, 0, 13 / 768, 0, 0, 0, 0],
                       [899 / 430080, 0, 61 / 15360, 0, 0, 0, 0, 0, 0],
                       [49561 / 41287680, 0, 0, 0, 0, 0, 0, 0, 0]])
    else:
        print("Error generating coefficients...")

    c = np.zeros([np.shape(c0)[0], 1])
    for i in range(0, np.shape(c0)[0]):
        c[i] = np.polyval(c0[i, :], e)

    return c


def create_raster_from_ogr_source(source, no_data_value=2, pixel_size=0.5, x_res= None, y_res=None):
    """
    Converts a geojson vector product into a raster stored as a numpy array
    :param source: OGR Geometry
    :param no_data_value: No data value to label pixels with no data. Default= 0
    :param pixel_size: pixel size in meters
    :return: numpy array
    """
    gdal.PushErrorHandler(gdal_error_handler)

    source_layer = source.GetLayer()

    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    if x_res is None or y_res is None:
        x_res = int((x_max-x_min) / pixel_size)
        y_res = int((y_max-y_min) / pixel_size)

    # Create the destination data source for index image
    ndx_target = gdal.GetDriverByName('GTiff').Create('SanCristobalCurrent.tif', x_res, y_res, 1, gdal.GDT_Byte)
    ndx_target.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    ndx_band = ndx_target.GetRasterBand(1)
    ndx_band.SetNoDataValue(0)
    # TODO: Correctly uses BuildingID, but doesn't burn with BuildingID, uses unique integers instead,
    #       which is probably fine, but it also doens't match the number of total features...
    gdal.RasterizeLayer(ndx_target, [1], source_layer, burn_values=[1])
    #ndx_raster = ndx_band.ReadAsArray().astype(np.uint16)

    # Create the destination data source for cls image
    # cls_target = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
    # cls_target.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    # cls_band = cls_target.GetRasterBand(1)
    # cls_band.SetNoDataValue(no_data_value)  # TODO: Doesn't work?
    # gdal.RasterizeLayer(cls_target, [1], source_layer, burn_values=[6])
    # cls_raster = cls_band.ReadAsArray().astype(np.uint8)
    # cls_raster[cls_raster != 0] = 6
    # cls_raster[cls_raster == 0] = 2

    # return ndx_raster, cls_raster, x_res, y_res


# todo: this function should really be refactored to share more code with the function above
def create_tiff_from_ogr_source(source, output_path, no_data_value=0, x_size=2048, y_size=2048, res=0.5):
    """
    Converts a geojson vector product into a raster stored as a numpy array
    :param source: OGR Geometry
    :param source_raster_path: Geotiff path to copy geo-reference metadata from
    :param no_data_value: No data value to label pixels with no data. Default= 0
    :param x_size: The width of the output raster
    :param y_size: The width of the output raster
    :return: numpy array
    """

    options = gdal.TranslateOptions(format='GTiff', width=x_size, height=y_size, projWinSRS='EPSG:4326',
                                    projWin=[lon_min, lat_max, lon_max, lat_min], bandList=[1, 2, 3])
    if not output_tiff_path.parent.exists():
        output_tiff_path.parent.mkdir()
    if not xml_src_path.exists():
        raise FileNotFoundError(xml_src_path.absolute())
    res = gdal.Translate(destName=str(output_tiff_path), srcDS=str(xml_src_path), options=options)
    # todo: remove print statements
    print(res)

    ##### **************************************************************************************************************
    source_layer = source.GetLayer()
    geo_transform = None
    projection = None
    if source_raster_path is not None:
        source_raster = gdal.Open(source_raster_path)  # needs to be a geotiff
        x_size = source_raster.RasterXSize
        y_size = source_raster.RasterYSize
        geo_transform = source_raster.GetGeoTransform()
        projection = source_raster.GetProjection()

    # Create the destination data source
    target = gdal.GetDriverByName('GTiff').Create('', x_size, y_size, 1, gdal.GDT_UInt16)

    # todo: this if statement may be unnecessary (gdal may allow setting things to None gracefully)
    if source_raster_path is not None:
        target.SetGeoTransform(geo_transform)
        target.SetProjection(projection)

    band = target.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    gdal.RasterizeLayer(target, [1], source_layer, options=["ATTRIBUTE=BuildingID"])
    raster = band.ReadAsArray()
    return raster


def create_geojson_from_raster(geotiff_path, geojson_filename):
    """
    Converts a geotiff to a geojson
    :param geotiff_path: Path to geotiff raster to be converted
    :param geojson_filename: Output filepath for geojson
    :return: None
    """
    raster = gdal.Open(geotiff_path)
    band = raster.GetRasterBand(1)
    driver = ogr.GetDriverByName("geojson")
    layer_name = "Polygonized"
    out_data = driver.CreateDataSource(geojson_filename)
    out_layer = out_data.CreateLayer(layer_name, srs=None)
    field_name = "Class"
    field = ogr.FieldDefn(field_name, ogr.OFTInteger)
    out_layer.CreateField(field)
    field_value = 0
    gdal.Polygonize(band, None, out_layer, field_value, [], callback=None)


def osm_to_geojson_file(osm_in):
    geo_json = osm_to_geojson(*parse_osm(osm_in))
    json_out = osm_in.with_suffix(".geojson")
    with open(json_out, "w") as out:
        json.dump(geo_json, out, indent=1)
    return json_out


if __name__ == "__main__":
    performer_path = Path(r"C:\Users\wangss1\Documents\Data\FFDA\OSM_Data\performer.osm")
    ground_truth_path=Path(r"C:\Users\wangss1\Documents\Data\FFDA\OSM_Data\ground_truth.osm")

    if not performer_path.exists():
        raise FileNotFoundError(performer_path.absolute())
    if not ground_truth_path.exists():
        raise FileNotFoundError(ground_truth_path.absolute())

    performer_geo = osm_to_geojson(*parse_osm(performer_path))
    ground_truth_geo = osm_to_geojson(*parse_osm(ground_truth_path))

    with open(performer_path.with_suffix('.geojson'), 'w') as geo_json_file:
        json.dump(performer_geo, geo_json_file)
    with open(ground_truth_path.with_suffix('.geojson'), 'w') as geo_json_file:
        json.dump(ground_truth_geo, geo_json_file)

    # TODO: HIGH PRIORITY
    # convert both to ogr
    ground_truth_ogr = osm_to_ogr(ground_truth_path)
    performer_ogr = osm_to_ogr(performer_path)

    # convert ground truth to raster
    # project performer into ground_truth raster

    print_once(dump_override=True)

