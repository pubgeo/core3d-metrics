import numpy as np


class Building:

    def __init__(self, label):
        """
        Constructor for building object
        :param label: unique building index
        """
        self.label = label
        self.match = False
        self.matched_with = None
        self.fp_overlap_with = []
        self.is_uncertain = False
        self.is_ignored = False
        self.has_error = False
        self.iou_score = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.points = None
        self.on_boundary = False
        self.area = None
        self.perimeter = None

    def calculate_area(self):
        """
        Calculates the pixel area of the building object by counting the number of points and stores it as part of the
        building object as well as returning the value
        :return:
        """
        self.area = len(self.points)
        return self.area

    def create_individual_building_raster(self):
        """
        Creates a minimized raster of a building object to calculate perimeter
        :return: raster of the building object
        """
        canvas = np.zeros((self.max_y - self.min_y + 1,
                           self.max_x - self.min_x + 1))
        for point in self.points:
            canvas[point[1] - self.min_y, point[0] - self.min_x] = 1
        return canvas


def create_raster_from_building_objects(building_list, x_res, y_res):
    """
    Creates a raster from a list of building objects.
    :param building_list: List of building objects
    :param x_res: X resolution of raster
    :param y_res: Y resolution of raster
    :return: raster of buildings
    """
    canvas = np.zeros((x_res, y_res))
    canvas = np.uint16(canvas)
    for current_building in building_list.items():
        for current_point in current_building[1].points:
            canvas[current_point[0], current_point[1]] = current_building[1].label
    return canvas
