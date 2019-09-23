import numpy as np
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv
from random import randint
import warnings
import functools


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'):
    """
    Call in a loop to create terminal progress bar
    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param length: character length of bar (Int)
    :param fill: bar fill character (Str)
    :return:
    """
    try:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
    except(ZeroDivisionError):
        print("Dividing by zero, check loop parameters...")

def calculate_metrics_iterator(gt_buildings, gt_indx_raster, ignored_gt, perf_indx_raster, performer_buildings,
                               iou_threshold, metrics_container):

    """
    Runs metrics on list of ground truth and performer buildings
    :param gt_buildings: list of ground truth building objects
    :param gt_indx_raster: numpy array of the ground truth indexed raster. Used for stoplight chart
    :param ignored_gt: list of ignored ground truth indices
    :param perf_indx_raster: numpy array of the performer indexed raster. Used for stoplight chart
    :param performer_buildings: list of performer building objects
    :param metrics_container: metrics container object to store the metrics
    :param iou_threshold: IOU threshold for passing grade
    :return: metrics container containing all the metrics
    """
    # Iterate through performer buildings and calculate iou
    matched_performer_indices = []
    ignored_performer = []
    fp_indices = []
    all_perimeter_diff = []
    all_perimeter_ratio = []
    all_area_diff = []
    all_area_ratio = []
    ignore_threshold = 0.5 # parser arg
    print("Iterating through performer buildings")
    TP = 0
    FP = 0
    print("Total performer buildings:" + repr(performer_buildings.__len__()))
    print("Total ground truth buildings:" + repr(gt_buildings.__len__()))
    iterations = 0
    total_iterations = len(performer_buildings.items())
    print_progress_bar(0, total_iterations, prefix='Progress:', suffix='Complete', length=50)
    for _, current_perf_building in performer_buildings.items():
        iterations = iterations+1
        print_progress_bar(iterations, total_iterations, prefix='Progress:', suffix='Complete', length=50)
        if current_perf_building.is_ignored is True:
            continue
        for _, current_gt_building in gt_buildings.items():
            # Calculate IOU
            iou, intersection, union = MetricsCalc.calculate_iou(current_gt_building, current_perf_building)
            # Check if Perf overlaps too much with ignored GT
            if current_gt_building.is_ignored is True:
                if intersection > ignore_threshold * current_perf_building.area:
                    current_perf_building.is_ignored = True
                    current_perf_building.match = True
                    ignored_performer.append(current_perf_building.label)
                    break

            if iou >= iou_threshold:
                TP = TP + 1
                matched_performer_indices.append(current_perf_building.label)
                current_gt_building.match = True
                current_gt_building.matched_with = current_perf_building.label
                current_perf_building.match = True
                current_perf_building.matched_with = current_gt_building.label
                all_perimeter_diff.append(MetricsCalc.calculate_perimeter_diff(
                    current_gt_building.perimeter, current_perf_building.perimeter))
                all_perimeter_ratio.append(MetricsCalc.calculate_perimeter_ratio(
                    current_gt_building.perimeter, current_perf_building.perimeter))
                all_area_diff.append(MetricsCalc.calculate_area_diff(
                    current_gt_building.area, current_perf_building.area))
                all_area_ratio.append(MetricsCalc.calculate_area_ratio(
                    current_gt_building.area, current_perf_building.area))
                break
            elif 0 < iou < iou_threshold:
                current_gt_building.fp_overlap_with.append(current_perf_building.label)
                current_perf_building.fp_overlap_with.append(current_gt_building.label)

        # If after going through all GT buildings and no match, its a FP
        if current_perf_building.match is False:
            fp_indices.append(current_perf_building.label)
            FP = FP + 1

    # all remaining unmatched GT are false negatives
    fn_indices = [idx for idx in gt_buildings if gt_buildings[idx].match is False]
    FN = len(fn_indices)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    print("Generating Stoplight Chart...")
    stoplight_generator = TileEvaluator()
    stoplight = stoplight_generator.generate_stoplight_chart(gt_indx_raster, perf_indx_raster,
                                                             matched_performer_indices, fn_indices, fp_indices,
                                                             ignored_gt, ignored_performer)
    metrics_container.set_values(TP, FP, FN, ignored_gt, ignored_performer, precision, recall, f1_score, None,
                                 fn_indices, matched_performer_indices, fp_indices, np.mean(all_area_diff),
                                 np.mean(all_area_ratio), stoplight)
    return metrics_container


def eval_instance_metrics(gt_cls_raster, gt_indx_raster, params, perf_indx_raster):
    start_time = time.time()
    edge_x, edge_y = np.shape(gt_indx_raster)
    # Get unique indices
    print("Getting unique indices in ground truth and performer files...")
    unique_performer_ids = np.unique(perf_indx_raster)
    unique_gt_ids = np.unique(gt_indx_raster)
    # Uncertain GT regions
    print("Getting uncertain/ignore areas...")
    uncertain_mask = np.zeros(np.shape(gt_cls_raster))
    uncertain_mask[gt_cls_raster == params.UNCERTAIN_VALUE] = 1
    uncertain_mask_indexed = np.multiply(gt_indx_raster, uncertain_mask)
    uncertain_indices = np.unique(uncertain_mask_indexed)
    uncertain_indices = [x for x in uncertain_indices if x != 0]
    # GCreate ground truth building objects
    print("Creating ground truth building objects...")
    gt_buildings = {}
    ignored_gt = []
    iterations = 0
    total_iterations = len(unique_gt_ids)
    print_progress_bar(0, total_iterations, prefix='Progress:', suffix='Complete', length=50)
    for current_index in unique_gt_ids:
        iterations = iterations + 1
        print_progress_bar(iterations, total_iterations, prefix='Progress:', suffix='Complete', length=50)
        print(str(iterations) + " out of " + str(total_iterations))
        if current_index == 0:
            continue
        current_building = Building(current_index)
        # Get x,y points of building pixels
        x_points, y_points = np.where(gt_indx_raster == current_index)
        if len(x_points) == 0 or len(y_points) == 0:
            continue
        # Get minimum and maximum points
        current_building.min_x = x_points.min()
        current_building.min_y = y_points.min()
        current_building.max_x = x_points.max()
        current_building.max_y = y_points.max()
        current_building.points = np.array(list(zip(x_points, y_points)))
        building_raster = MetricsCalc.create_individual_building_raster(current_building)
        # Calculate Perimeter
        current_building.perimeter = MetricsCalc.calculate_perimeter(building_raster)
        # Calculate Area
        current_building.area = MetricsCalc.calculate_area(current_building)
        # Create dictionary entry based on building index
        gt_buildings[current_index] = current_building
        # Check if building is on the edge of AOI
        if current_building.min_x == 0 or current_building.min_y == 0 or current_building.max_x == edge_x \
                or current_building.max_y == edge_y:
            current_building.on_boundary = True
        # TODO: Filter by area, not only minimum size
        if current_building.area < params.MIN_AREA_FILTER:
            current_building.is_ignored = True
            ignored_gt.append(current_building.label)
        if current_building.label in uncertain_indices:
            current_building.is_uncertain = True
            current_building.is_ignored = True
            ignored_gt.append(current_building.label)
    ignored_gt = list(np.unique(ignored_gt))
    # Create performer building objects
    print("Creating performer building objects...")
    performer_buildings = {}
    iterations = 0
    total_iterations = len(unique_performer_ids)
    print_progress_bar(0, total_iterations, prefix='Progress:', suffix='Complete', length=50)
    for current_index in unique_performer_ids:
        iterations = iterations + 1
        print_progress_bar(iterations, total_iterations, prefix='Progress:', suffix='Complete', length=50)
        print(str(iterations) + " out of " + str(total_iterations))
        if current_index == 0:
            continue
        current_building = Building(current_index)
        # Get x,y points of building pixels
        x_points, y_points = np.where(perf_indx_raster == current_index)
        if len(x_points) == 0 or len(y_points) == 0:
            continue
        # Get minimum and maximum points
        current_building.min_x = x_points.min()
        current_building.min_y = y_points.min()
        current_building.max_x = x_points.max()
        current_building.max_y = y_points.max()
        current_building.points = np.array(list(zip(x_points, y_points)))
        building_raster = MetricsCalc.create_individual_building_raster(current_building)
        # Calculate Perimeter
        current_building.perimeter = MetricsCalc.calculate_perimeter(building_raster)
        # Calculate Area
        current_building.area = MetricsCalc.calculate_area(current_building)
        # Create dictionary entry based on building index
        performer_buildings[current_index] = current_building
        # Check if building is on the edge of AOI
        if current_building.min_x == 0 or current_building.min_y == 0 or current_building.max_x == edge_x \
                or current_building.max_y == edge_y:
            current_building.on_boundary = True
    # Create a metrics container
    metrics_container_no_merge = MetricsContainer()
    metrics_container_no_merge.name = "No Merge"
    # Calculate Metrics
    print("Calculating metrics without merge...")
    metrics_container_no_merge = calculate_metrics_iterator(gt_buildings, gt_indx_raster, ignored_gt, perf_indx_raster,
                                                            performer_buildings, params.IOU_THRESHOLD,
                                                            metrics_container_no_merge)
    metrics_container_no_merge.show_metrics()
    # metrics_container_no_merge.show_stoplight_chart()
    # Merge performer buildings
    print("Merging FP buildings to account for closely spaced buildings...")
    merged_performer_buildings = merge_false_positives(edge_x, edge_y, gt_buildings, performer_buildings)
    # Create new performer_index raster
    canvas = create_raster_from_building_objects(merged_performer_buildings, perf_indx_raster.shape[0],
                                                 perf_indx_raster.shape[1])
    # Reset matched flags in gt_buildings
    for _, current_gt_building in gt_buildings.items():
        current_gt_building.match = False
        current_gt_building.fp_overlap_with = []
        current_gt_building.iou_score = None
        current_gt_building.is_uncertain = False
    # Re-run metrics on merged performer buildings
    metrics_container_merge_fp = MetricsContainer()
    metrics_container_merge_fp.name = "Merge FP Performers"
    metrics_container_merge_fp = calculate_metrics_iterator(gt_buildings, gt_indx_raster, ignored_gt, canvas,
                                                            merged_performer_buildings, params.IOU_THRESHOLD,
                                                            metrics_container_merge_fp)
    metrics_container_merge_fp.show_metrics()
    # metrics_container_merge_fp.show_stoplight_chart()
    # Merge gt buildings
    print("Merging GT buildings to account for closely spaced buildings...")
    # merged_gt_buildings = merge_false_negatives(edge_x, edge_y, gt_buildings, performer_buildings)
    merged_gt_buildings = merge_false_positives(edge_x, edge_y, performer_buildings, gt_buildings)
    canvas = create_raster_from_building_objects(merged_gt_buildings, gt_indx_raster.shape[0], gt_indx_raster.shape[1])
    for _, current_perf_building in performer_buildings.items():
        current_perf_building.match = False
        current_perf_building.fp_overlap_with = []
        current_perf_building.iou_score = None
        current_perf_building.is_uncertain = False
    metrics_container_merge_fn = MetricsContainer()
    metrics_container_merge_fn.name = "Merge GTs"
    metrics_container_merge_fn = calculate_metrics_iterator(merged_gt_buildings, gt_indx_raster, ignored_gt, canvas,
                                                            performer_buildings, params.IOU_THRESHOLD,
                                                            metrics_container_merge_fn)
    metrics_container_merge_fn.show_metrics()
    # metrics_container_merge_fn.show_stoplight_chart()
    elapsed_time = time.time() - start_time
    print("Elapsed time: " + repr(elapsed_time))
    print("Done")


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
        yellow = [0, 255, 255]
        white = [255, 255, 255]
        black = [0, 0, 0]
        # true positives
        for i in tp_indices:
            stoplight_chart[perf == i] = green
        # false negatives
        for i in fn_indices:
            stoplight_chart[gt == i] = red
        # false positives
        for i in fp_indices:
            stoplight_chart[perf == i] = yellow
        # ignored instances
        for i in ignore_gt_indices:
            stoplight_chart[gt == i] = white
        for i in ignore_perf_indices:
            stoplight_chart[perf == i] = white
        # uncertain instances
        if uncertain_mask is not None:
            stoplight_chart[uncertain_mask == 1] = white
        # get contours of ground truth buildings
        ret, threshed = cv.threshold(gt, 0, 2 ** 16, cv.THRESH_BINARY)
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