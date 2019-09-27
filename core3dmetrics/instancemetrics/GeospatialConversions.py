import numpy as np
from osgeo import gdal


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


if __name__ == "__main__":
    print('Debug')

