import numpy as np
from core3dmetrics.instancemetrics.TileEvaluator import TileEvaluator, merge_buildings
import time as time
from core3dmetrics.instancemetrics.Building_Classes import Building, create_raster_from_building_objects
from core3dmetrics.instancemetrics.MetricsCalculator import MetricsCalculator as MetricsCalc
from core3dmetrics.instancemetrics.MetricsContainer import MetricsContainer


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
    iou_per_gt_building = {}
    matched_performer_indices = []
    ignored_performer = []
    fp_indices = []
    all_perimeter_diff = []
    all_perimeter_ratio = []
    all_area_diff = []
    all_area_ratio = []
    ignore_threshold = 0.5  # parser arg
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
                # Record IOU of matched GT building
                iou_per_gt_building[current_gt_building.label] = [iou, [np.average(current_gt_building.points[:, 0]),
                                                                        np.average(current_gt_building.points[:, 1])]]
                # Do not let multiple matches with one GT building
                if current_gt_building.match is True:
                    break
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
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision+recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    print("Generating Stoplight Chart...")
    stoplight_generator = TileEvaluator()
    stoplight = stoplight_generator.generate_stoplight_chart(gt_indx_raster, perf_indx_raster,
                                                             matched_performer_indices, fn_indices, fp_indices,
                                                             ignored_gt, ignored_performer)
    metrics_container.set_values(TP, FP, FN, ignored_gt, ignored_performer, precision, recall, f1_score, None,
                                 fn_indices, matched_performer_indices, fp_indices, np.mean(all_area_diff),
                                 np.mean(all_area_ratio), stoplight, iou_per_gt_building)
    return metrics_container


def eval_instance_metrics(gt_indx_raster, params, perf_indx_raster):
    start_time = time.time()
    edge_x, edge_y = np.shape(gt_indx_raster)
    # Get unique indices
    print("Getting unique indices in ground truth and performer files...")
    unique_performer_ids = np.unique(perf_indx_raster)
    unique_gt_ids = np.unique(gt_indx_raster)

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
        # print(str(iterations) + " out of " + str(total_iterations))
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
        # print(str(iterations) + " out of " + str(total_iterations))
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

    elapsed_time = time.time() - start_time
    print("Elapsed time: " + repr(elapsed_time))
    return metrics_container_no_merge

