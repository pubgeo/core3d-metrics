import numpy as np


class MetricsCalculator:

    def __init__(self):
        self.iou_threshold = 0.5

    @staticmethod
    def calculate_iou(ground_truth_building, performer_building):
        """
        Calculates the intersection over union (IOU) of a 2 Building objects
        :param ground_truth_building: ground truth building object
        :param performer_building: performer building object
        :return: IOU, intersection, and union
        """
        if ground_truth_building.min_x > performer_building.max_x or \
                ground_truth_building.min_y > performer_building.max_y or \
                ground_truth_building.max_x < performer_building.min_x or \
                ground_truth_building.max_y < performer_building.min_y:
            return 0, 0, 0
        if len(ground_truth_building.points) == 0 or len(performer_building.points) == 0:
            return 0, 0, 0
        ground_truth_points_set = set([tuple(x) for x in ground_truth_building.points])
        performer_points_set = set([tuple(x) for x in performer_building.points])
        common = ground_truth_points_set.intersection(performer_points_set)
        intersection = len(common)
        union = len(performer_building.points) + len(ground_truth_building.points) - intersection
        return intersection/union, intersection, union

    @staticmethod
    def calculate_perimeter(polygon_raster, pixel_size=1):
        """
        Calculates perimeter of a solid raster polygon using improved
        algorithm found here: http://www.geocomputation.org/1999/076/gc_076.htm
        :param polygon_raster: binary raster of shape to be calculated
        :param pixel_size: size of pixel, default 1. This number is multiplied to the pixel perimeter
        :return: perimeter of shape
        """
        key_set = list(range(0, 256))
        value_set = [4, 4, 3, np.sqrt(2), 4, 4, np.sqrt(2), 2 * np.sqrt(2), 3, np.sqrt(2), 2, 2, 3, np.sqrt(2),
                     np.sqrt(2), np.sqrt(2), 3, 3, 2, np.sqrt(2), np.sqrt(2), np.sqrt(2), 2, np.sqrt(2), 2, 2,
                     1, 1, 2, 2, 1, 1, 4, 4, 3, np.sqrt(2), 4, 4, 3, 3, np.sqrt(2), 2 * np.sqrt(2), np.sqrt(2),
                     np.sqrt(2),
                     3,
                     3, np.sqrt(2), np.sqrt(2), 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 1, np.sqrt(2), 2, 2, 1, 1,
                     3, 3, 2, 2, 3, 3, 2, 2, 2, np.sqrt(2), 1, 1, 2, 2, 1, np.sqrt(2), 2, 2, 1, 1, np.sqrt(2), 2, 1,
                     np.sqrt(2), 1, 1, 0, 0, 1, 2, 0, 0, np.sqrt(2), np.sqrt(2), 2, 2, 3, 3, 2, 2, 2, np.sqrt(2), 1, 1,
                     2, 2, 1, 1, np.sqrt(2), 2, 1, 2, np.sqrt(2), np.sqrt(2), 1, 1, 1, np.sqrt(2), 0, 0, 1, 1, 0, 0,
                     4, 4, 3, 3, 4, 4, np.sqrt(2), 3, 3, 3, 2, 2, 3, 3, 2, 2, np.sqrt(2), 3, np.sqrt(2), np.sqrt(2),
                     2 * np.sqrt(2), 3, np.sqrt(2), np.sqrt(2), 2, 2, 1, 1, 2, 2, np.sqrt(2), 1, 4, 4, 3, 3, 4, 4, 3, 3,
                     np.sqrt(2), 3, 2, 2, 3, 3, np.sqrt(2), np.sqrt(2), np.sqrt(2), 3, 2, np.sqrt(2), 3, 3, 2,
                     np.sqrt(2),
                     2, 2, 2, 1, 2, np.sqrt(2), 1, 1, np.sqrt(2), 3, 2, 2, np.sqrt(2), 3, 2, 2, np.sqrt(2), np.sqrt(2),
                     1,
                     1, 2, np.sqrt(2), 2, 1, 2, 2, 1, 1, np.sqrt(2), 2, 1, 1, 1, 1, 0, 0, np.sqrt(2), 1, 0, 0,
                     2 * np.sqrt(2), 3, 2, 2, 3, 3, 2, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2), 1, 2, np.sqrt(2),
                     1, 1, np.sqrt(2), 2, np.sqrt(2), 1, np.sqrt(2), np.sqrt(2), 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        # Define lookup table
        pattern_table = dict(zip(key_set, value_set))
        # Define pixel algorithm mask
        perimeter_code_window = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])

        # Pad the raster with single row of 0s
        polygon_raster = np.pad(polygon_raster, 1, 'constant', constant_values=0)

        # Find all indices of pixel values
        row, col = np.nonzero(polygon_raster)
        perimeter_contribution = np.zeros(row.__len__())
        for i in range(0, row.__len__()):
            boundary_window = polygon_raster[row[i]-1:row[i]+2, col[i]-1:col[i]+2]
            # Apply algorithm mask
            applied_mask = np.multiply(boundary_window, perimeter_code_window)
            perimeter_code = np.sum(applied_mask)
            perimeter_contribution[i] = pattern_table[perimeter_code]
        pixel_perimeter = np.sum(perimeter_contribution)*pixel_size
        return pixel_perimeter

    @staticmethod
    def create_individual_building_raster(building_object):
        """
        Creates a minimized raster of a building object to calculate perimeter
        :param building_object: building object to be rastrized
        :return: raster of the building object
        """
        canvas = np.zeros((building_object.max_y - building_object.min_y + 1,
                           building_object.max_x - building_object.min_x + 1))
        for point in building_object.points:
            canvas[point[1] - building_object.min_y, point[0] - building_object.min_x] = 1
        return canvas

    @staticmethod
    def calculate_area(building, pixel_size=1):
        """
        Calculates the area of a building object in the corresponding pixel_size measurement
        :param building: building object to be evaluated
        :param pixel_size: pixel size, default set to 1
        :return: Area of the building. Without providing a pixel size, it will just return the area in pixels
        """
        return len(building.points) * (pixel_size**2)

    @staticmethod
    def calculate_perimeter_ratio(gt_perimeter, perf_perimeter):
        """
        Calculates the perimeter ratio between two perimeters
        :param gt_perimeter: perimeter of ground truth building
        :param perf_perimeter: perimeter of performer building
        :return: perimeter ratio
        """
        return min(gt_perimeter, perf_perimeter) / max(gt_perimeter, perf_perimeter)

    @staticmethod
    def calculate_area_ratio(gt_area, perf_area):
        """
        Calculates the area ratio metric
        :param gt_area: area of ground truth building
        :param perf_area: area of performer building
        :return: area ratio
        """
        return min(gt_area, perf_area) / max(gt_area, perf_area)

    @staticmethod
    def calculate_perimeter_diff(gt_perim, perf_perim):
        """
        Calculates the perimeter difference metric
        :param gt_perim: perimeter of ground truth building
        :param perf_perim: perimeter of performer building
        :return: perimeter difference
        """
        return abs(gt_perim - perf_perim) / gt_perim

    @staticmethod
    def calculate_area_diff(gt_area, perf_area):
        """
        Calculates the area difference metric
        :param gt_area: area of ground truth building
        :param perf_area: area of performer building
        :return: area difference metric
        """
        return abs(gt_area - perf_area) / gt_area

    @staticmethod
    def calculate_area_from_raster(polygon_raster, pixel_size=1):
        """
        Calculates area of a polygon from raster input
        :param polygon_raster: binary raster of shape to be calculated
        :param pixel_size: size of pixel, default 1. This number is multiplied to the pixel area
        :return: area of polygon
        """
        return np.sum(polygon_raster)*(pixel_size**2)
