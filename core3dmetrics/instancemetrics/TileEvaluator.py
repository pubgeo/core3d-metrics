from PIL import Image
import numpy as np
import cv2 as cv
from random import randint
import warnings
import functools
from .Building_Classes import Building
from .MetricsCalculator import MetricsCalculator


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class TileEvaluator:
    """
        Used to store and compute metric scores
    """
    def __init__(self):
        self.Image_path = None

    @staticmethod
    def read_image(image_path):
        """
        read in image from path
        """
        im = Image.open(image_path, 'r')
        return np.array(im)

    @staticmethod
    def im_show(image_path):
        """
        reads in image, thresholds it, and displays
        """
        img = cv.imread(image_path, cv.IMREAD_ANYDEPTH)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        ret, threshed = cv.threshold(img, 0, 2 ** 16, cv.THRESH_BINARY)
        print(ret)
        print(threshed.shape, threshed.dtype)
        cv.imshow('image', threshed)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def tabulate(image_path):
        """
        prints building instance, building size
        """
        img = cv.imread(image_path, cv.IMREAD_ANYDEPTH)
        unique, counts = np.unique(img, return_counts=True)
        dummy = [print(a[0], a[1]) for a in zip(unique, counts)]

    @staticmethod
    def get_instance_contours(img, contoured, instance):
        """
        get the contour for a specific building instance and draw it
        """
        mask = np.zeros(img.shape, dtype=np.uint16)
        mask[img == instance] = 1
        ret, threshed = cv.threshold(mask, 0, 2 ** 16, cv.THRESH_BINARY)
        compressed = threshed.astype(np.uint8)
        contours, hierarchy = cv.findContours(compressed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(contoured, contours, -1, (randint(25, 255), randint(25, 255), randint(25, 255)), 3)
        img2 = contours = hierarchy = mask = None

    @staticmethod
    def get_instance_bounding_box(img, bounding_boxes, instance):
        """
        get the bounding box for a specific building instance and draw it
        """
        mask = np.zeros(img.shape, dtype=np.uint16)
        mask[img == instance] = 1
        ret, threshed = cv.threshold(mask, 0, 2 ** 16, cv.THRESH_BINARY)
        compressed = threshed.astype(np.uint8)
        contours, hierarchy = cv.findContours(compressed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        x, y, w, h = cv.boundingRect(contours[0])
        cv.rectangle(bounding_boxes, (x, y), (x + w, y + h), (randint(25, 255), randint(25, 255), randint(25, 255)), 3)
        img2 = contours = hierarchy = mask = None

    def draw_contours(self, image_path):
        """
        for each building instance in the image, draw its contour in a different color
        """
        img = cv.imread(image_path, cv.IMREAD_ANYDEPTH)
        contoured = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        unique, counts = np.unique(img, return_counts=True)
        for uni in unique:
            if uni == 0:
                continue
            self.get_instance_contours(img, contoured, uni)

        cv.namedWindow('building contours', cv.WINDOW_NORMAL)
        cv.imshow('building contours', contoured)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_bounding_boxes(self, image_path):
        """
        for each building instance in the image, draw its bounding box in a different color
        """
        img = cv.imread(image_path, cv.IMREAD_ANYDEPTH)
        bboxes = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        unique, counts = np.unique(img, return_counts=True)
        for uni in unique:
            if uni == 0:
                continue
            self.get_instance_bounding_box(img, bboxes, uni)

        cv.namedWindow('building bounding boxes', cv.WINDOW_NORMAL)
        cv.imshow('building bounding boxes', bboxes)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @deprecated
    def merge_ambiguous_buildings(self, raster, kernel_size=4):
        """
        DEPRECATED - Use merge_false_positives in run_metrics.py.
        Naive approach to merging ambiguous buildings. Dilation+erode to remove spaces between buildings
        :param raster: raster (nd.array)
        :param kernel_size: merge kernel size (int)
        :return:
        """
        # Merge perf buildings that are close together using dilate and erode
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv.dilate(raster, kernel, iterations=1)
        erosion = cv.erode(dilation, kernel, iterations=1)
        # new building regions have multiple values within
        # turn into binary raster to get new building contours
        erosion[erosion > 0] = 1
        ret, threshed = cv.threshold(erosion, 0, 2 ** 16, cv.THRESH_BINARY)
        compressed = threshed.astype(np.uint8)
        contours, hierarchy = cv.findContours(compressed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        merged_erosion = np.zeros(erosion.shape, dtype=np.uint16)
        # refill contours with a single value
        for idx, c in enumerate(contours):
            cv.fillPoly(merged_erosion, [c], color=idx)
        return merged_erosion

    @staticmethod
    def get_num_instances(im, non_building_labels):
        """
        get list of unique building instances
        non_building_labels is an array of numbers that do not represent building instances (i.e. non-building or void)
        """
        return np.setdiff1d(im, non_building_labels)

    @staticmethod
    def get_current_building_mask(im, instance):
        """
        return binary mask with just the current building instance
        return its area (number of pixels)
        """
        current_building_mask = np.zeros(im.shape, dtype=np.uint16)
        current_building_mask[im == instance] = 1
        current_building_area = np.sum(current_building_mask)
        return current_building_mask, current_building_area

    def filter_instances_by_size(self, im, unique_instances, min_building_size):
        """
        filters all instances in a single image by size
        returns a list of instances to keep evaluating and a list of instances to ignore
        """
        # create array to store building instances to ignore
        ignored_instances = np.array([])
        # if min_building_size is negative, error
        if min_building_size < 0:
            raise ValueError("Building size filter cannot be a negative number")
        # return list of instances to check and list of instances to ignore
        # if min_building_size is 0, return original array of instances, ignored_instances is empty
        if min_building_size == 0:
            return unique_instances, ignored_instances
        else:
            for i in range(len(unique_instances)):
                _, current_building_size = self.get_current_building_mask(im, unique_instances[i])
                if current_building_size < min_building_size:
                    ignored_instances = np.append(ignored_instances, i)
            return np.setdiff1d(unique_instances, ignored_instances), ignored_instances

    def filter_edge_instances(self, im, current_instance, min_building_size, unique_instances, ignored_instances):
        """
        check if current building is on edge and also too small
        if so, ignore this instance
        updates the keep/ignore lists
        """
        max_x = im.shape[0] - 1
        max_y = im.shape[1] - 1
        current_building_mask, current_building_size = self.get_current_building_mask(im, current_instance)
        # get indices of nonzero elements in current_building_mask
        row, col = np.nonzero(current_building_mask)
        if np.any(row == max_x) or np.any(col == max_y):
            if current_building_size < min_building_size:
                # revise lists if found a new instance to ignore
                ignored_instances = np.append(ignored_instances, current_instance)
                unique_instances = np.setdiff1d(unique_instances, current_instance)
        return unique_instances, ignored_instances

    @staticmethod
    def get_building_contour(current_building_mask):
        """
        from current_building_mask, return its contour
        """
        ret, threshed = cv.threshold(current_building_mask, 0, 2 ** 16, cv.THRESH_BINARY)
        compressed = threshed.astype(np.uint8)
        current_building_contour, hierarchy = cv.findContours(compressed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return current_building_contour, hierarchy

    @staticmethod
    def get_bounding_box(current_building_contour):
        """
        from current_building_contour, get its rectangular, non-rotated bounding box coordinates
        """
        x, y, w, h, = cv.boundingRect(current_building_contour[0])
        return x, y, w, h

    @staticmethod
    def crop_bounding_box(im, x, y, w, h):
        """
        crop image using pixel coordinates
        will be used to get corresponding instances in same pixel locations of two images that we compare
        """
        return im[y:y+h, x:x+w]

    @staticmethod
    def compute_pixel_iou(perf_building_mask, gt_building_mask):
        """
        computes IoU between performer and ground truth building masks
        """
        if perf_building_mask.shape != gt_building_mask.shape:
            raise ValueError("Dimension mismatch")
        intersection = np.sum(perf_building_mask & gt_building_mask)
        union = np.sum(perf_building_mask | gt_building_mask)
        iou = intersection / union
        return iou

    @staticmethod
    def generate_stoplight_chart(gt, perf, tp_indices, fn_indices, fp_indices, ignore_gt_indices,
                                 ignore_perf_indices, uncertain_mask=None):
        if gt.shape != perf.shape:
            raise ValueError("Dimension mismatch")
        stoplight_chart = np.multiply(np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.uint8), 220)
        green = [0, 255, 0]
        red = [0, 0, 255]
        blue = [255, 0, 0]
        yellow = [0, 255, 255]
        white = [255, 255, 255]
        black = [0, 0, 0]
        # true positives
        for i in tp_indices:
            stoplight_chart[perf == i] = white
        # false negatives
        for i in fn_indices:
            stoplight_chart[gt == i] = blue
        # false positives
        for i in fp_indices:
            stoplight_chart[perf == i] = red
        # ignored instances
        for i in ignore_gt_indices:
            stoplight_chart[gt == i] = yellow
        for i in ignore_perf_indices:
            stoplight_chart[perf == i] = yellow
        # uncertain instances
        if uncertain_mask is not None:
            stoplight_chart[uncertain_mask == 1] = white
        # get contours of ground truth buildings
        gt_binary = gt.copy()
        gt_binary[gt != 0] = 1
        gt_binary = gt_binary.astype(np.uint8)
        ret, threshed = cv.threshold(gt_binary, 0, 255, cv.THRESH_BINARY)
        compressed = threshed.astype(np.uint8)
        contours, hierarchy = cv.findContours(compressed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        cv.drawContours(stoplight_chart, contours, -1, black, 2)
        return stoplight_chart

    def draw_iou_on_stoplight(self, stoplight_chart, iou, position):
        """
        :param stoplight_chart: stoplight chart
        :param iou: list of IoUs
        :param  position: list of positions of building centers to draw IoUs on
        :return: stoplight chart with IoUs on each building instance
        """
        iou = np.around(iou, decimals= 3)
        stoplight_with_iou = stoplight_chart.copy()
        for current_iou, current_position in list(zip(iou, position)):
            cv.putText(stoplight_with_iou, str(current_iou),
                       (current_position[0].astype('int'), current_position[1].astype('int')),
                       cv.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])
        stoplight_with_iou_rgb = self.bgr_to_rgb(stoplight_with_iou)
        return stoplight_with_iou, stoplight_with_iou_rgb

    @staticmethod
    def bgr_to_rgb(bgr_image):
        rgb_image = bgr_image[..., ::-1]
        return rgb_image


def merge_buildings(edge_x, edge_y, gt_buildings, performer_buildings):
    """
    Merges the overlapping buildings from 2 sets of building objects that have already been run through metrics and
    outputs a new list of merged buildings. This makes evaluation more fair if performer buildings had many
    closely spaced buildings that all individually failed to meet the IOU threshold
    :param edge_x: maximum horizontal resolution
    :param edge_y: maximum vertical resolution
    :param gt_buildings: List of ground truth building objects that have already been run through metrics
    :param performer_buildings: List of performer building objects that have already been run against the ground truth
    buildings provided
    :return: list of merged performer buildings
    """
    metrics_calc = MetricsCalculator()
    # Merge Performer Buildings
    merged_performer_buildings = {}
    used_indices = []
    for _, current_perf_building in performer_buildings.items():
        if current_perf_building.label in used_indices:
            continue
        current_merged_building = Building(current_perf_building.label)
        if current_perf_building.fp_overlap_with.__len__() == 0:
            merged_performer_buildings[current_merged_building.label] = current_perf_building
        else:
            if current_perf_building.fp_overlap_with.__len__() > 1:
                print('More than one fp overlap, defaulting to first one...')
            fp_gt_building = gt_buildings[current_perf_building.fp_overlap_with[0]]
            for i in fp_gt_building.fp_overlap_with:
                used_indices.append(i)
                if current_merged_building.points is None:
                    current_merged_building.points = performer_buildings[i].points
                else:
                    current_merged_building.points = np.concatenate((current_merged_building.points,
                                                                     performer_buildings[i].points), axis=0)
            # Calculate merged building statistics
            current_merged_building.min_x = min(current_merged_building.points[:, 0])
            current_merged_building.min_y = min(current_merged_building.points[:, 1])
            current_merged_building.max_x = max(current_merged_building.points[:, 0])
            current_merged_building.max_y = max(current_merged_building.points[:, 1])
            # Calculate Perimeter
            # TODO: Figure out what to do about perimeter
            current_merged_building.perimeter = 1
            # Calculate Area
            current_merged_building.area = metrics_calc.calculate_area(current_merged_building)
            # Create dictionary entry based on building index
            merged_performer_buildings[current_merged_building.label] = current_merged_building
            # Check if building is on the edge of AOI
            if current_merged_building.min_x == 0 or current_merged_building.min_y == 0 or \
                    current_merged_building.max_x == edge_x or current_merged_building.max_y == edge_y:
                current_merged_building.on_boundary = True
    return merged_performer_buildings


def main():
    print("Debug")


if __name__ == "__main__":
    main()
