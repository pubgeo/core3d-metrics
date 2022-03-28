
import numpy as np

import scipy.ndimage as ndimage
import time
import multiprocessing
from .metrics_util import getUnitWidth
from .threshold_geometry_metrics import run_threshold_geometry_metrics
from .relative_accuracy_metrics import run_relative_accuracy_metrics
from core3dmetrics.instancemetrics.instance_metrics import eval_instance_metrics


def eval_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, plot=None, testCONF=None,
                 verbose=True):

    # Evaluate threshold geometry metrics using refDTM as the testDTM to mitigate effects of terrain modeling
    # uncertainty
    result_geo, unitArea, _, _ = run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, refDTM, testMask, tform, ignoreMask,
                                                plot=plot, for_objectwise=True, testCONF=testCONF, verbose=verbose)

    # Run the relative accuracy metrics and report results.
    result_acc = run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask,
                                               getUnitWidth(tform), for_objectwise=True, plot=plot)

    return result_geo, result_acc, unitArea


# Compute statistics on a list of values
def metric_stats(val):
    s = dict()
    s['values'] = val.tolist()
    s['mean'] = np.mean(val)
    s['stddev'] = np.std(val)
    s['pctl'] = {}
    s['pctl']['rank'] = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 91, 92, 93, 94, 95, 96, 96, 98, 99, 100]
    try:
        s['pctl']['value'] = np.percentile(val, s['pctl']['rank']).tolist()
    except IndexError:
        s['pctl']['value'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    return s


def multiprocessing_fun(ref_ndx, loop_region, refMask, test_ndx, ref_ndx_orig,
                        ref_use_counter, testMask, test_use_counter, refDSM,
                        refDTM, testDSM, testDTM, tform,
                        ignoreMask, plot, verbose, max_area, min_area,
                        max_volume, min_volume):
    # Reference region under evaluation
    ref_objs = (ref_ndx == loop_region) & refMask

    # Find test regions overlapping with ref
    test_regions = np.unique(test_ndx[ref_ndx == loop_region])

    # Find test regions overlapping with ref
    ref_regions = np.unique(ref_ndx_orig[ref_ndx == loop_region])

    # Remove background region, '0'
    if np.any(test_regions == 0):
        test_regions = test_regions.tolist()
        test_regions.remove(0)
        test_regions = np.array(test_regions)

    if np.any(ref_regions == 0):
        ref_regions = ref_regions.tolist()
        ref_regions.remove(0)
        ref_regions = np.array(ref_regions)

    if len(test_regions) == 0:
        return None

    for refRegion in ref_regions:
        # Increment counter for ref region used
        ref_use_counter[refRegion - 1] = ref_use_counter[refRegion - 1] + 1

    # Make mask of overlapping test regions
    test_objs = np.zeros_like(testMask)
    for test_region in test_regions:
        test_objs = test_objs | (test_ndx == test_region)
        # Increment counter for test region used
        test_use_counter[test_region - 1] = test_use_counter[test_region - 1] + 1

    # TODO:  Not practical as implemented to enable plots. plots is forced to false.
    [result_geo, result_acc, unitArea] = eval_metrics(refDSM, refDTM, ref_objs, testDSM, testDTM, test_objs, tform,
                                                      ignoreMask, plot=plot, verbose=verbose)

    this_metric = dict()
    this_metric['ref_objects'] = ref_regions.tolist()
    this_metric['test_objects'] = test_regions.tolist()
    this_metric['threshold_geometry'] = result_geo
    this_metric['relative_accuracy'] = result_acc

    # Calculate min and max area/volume
    if this_metric['threshold_geometry']['area']['test_area'] > max_area or loop_region == 1:
        max_area = this_metric['threshold_geometry']['area']['test_area']
    if this_metric['threshold_geometry']['area']['test_area'] < min_area or loop_region == 1:
        min_area = this_metric['threshold_geometry']['area']['test_area']
    if this_metric['threshold_geometry']['volume']['test_volume'] > max_volume or loop_region == 1:
        max_volume = this_metric['threshold_geometry']['volume']['test_volume']
    if this_metric['threshold_geometry']['volume']['test_volume'] < min_volume or loop_region == 1:
        min_volume = this_metric['threshold_geometry']['volume']['test_volume']

    return this_metric, result_geo, result_acc, unitArea, ref_regions


def run_objectwise_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, merge_radius=2,
                           plot=None, verbose=True, geotiff_filename=None, use_multiprocessing=False):

    # parse plot input
    if plot is None:
        PLOTS_ENABLE = False
    else:
        PLOTS_ENABLE = True
        PLOTS_SAVE_PREFIX = "objectwise_"

    # Number of pixels to dilate reference object mask
    padding_pixels = np.round(merge_radius / getUnitWidth(tform))
    strel = ndimage.generate_binary_structure(2, 1)

    # Dilate reference object mask to combine closely spaced objects
    ref_ndx_orig = np.copy(refMask)
    #ref_ndx = ndimage.binary_dilation(ref_ndx_orig, structure=strel,  iterations=padding_pixels.astype(int))
    ref_ndx = ref_ndx_orig

    # Create index regions
    ref_ndx, num_ref_regions = ndimage.label(ref_ndx)
    ref_ndx_orig, num_ref_regions = ndimage.label(ref_ndx_orig)
    test_ndx, num_test_regions = ndimage.label(testMask)

    # Get Height from DSM-DTM
    ref_height = refDSM.astype(np.float64) - refDTM.astype(np.float64)
    ref_height[ref_ndx == 0] = 0
    test_height = testDSM.astype(np.float64) - testDTM.astype(np.float64)
    test_height[test_ndx == 0] = 0

    # Keep track of how many times each region is used
    test_use_counter = np.zeros([num_test_regions, 1])
    ref_use_counter = np.zeros([num_ref_regions, 1])

    # Calculate instance metrics
    class instance_parameters:
        def __init__(self):
            self.IOU_THRESHOLD = 0.5
            self.MIN_AREA_FILTER = 0
            self.UNCERTAIN_VALUE = 65
    params = instance_parameters()
    metrics_container_no_merge = eval_instance_metrics(ref_ndx, params, test_ndx)
    no_merge_f1 = metrics_container_no_merge.f1_score
    num_buildings_performer = np.unique(test_ndx).__len__()-1
    num_buildings_truth = np.unique(ref_ndx).__len__()-1

    # Initialize metric list
    metric_list = []
    # Make images to visualize certain metrics
    # Unused regions will be marked as zero, background as -1
    image_out = refMask.astype(float).copy() - 1
    #image_out[image_out == -1] = np.nan
    image_2d_completeness = image_out.copy()
    image_2d_correctness = image_out.copy()
    image_2d_jaccard_index = image_out.copy()
    image_3d_completeness = image_out.copy()
    image_3d_correctness = image_out.copy()
    image_3d_jaccard_index = image_out.copy()
    image_hrmse = image_out.copy()
    image_zrmse = image_out.copy()

    # Initialize min/max area/volume
    max_area = 0
    min_area = 0
    max_volume = 0
    min_volume = 0

    # Create argument list
    arguments = []
    for loop_region in range(1, num_ref_regions + 1):
        arguments.append([ref_ndx, loop_region, refMask, test_ndx, ref_ndx_orig,
                            ref_use_counter, testMask, test_use_counter, refDSM,
                            refDTM, testDSM, testDTM, tform,
                            ignoreMask, plot, verbose, max_area, min_area,
                            max_volume, min_volume])

    if use_multiprocessing:
        with multiprocessing.Pool() as pool:
            result_map = pool.starmap(multiprocessing_fun, arguments)
            pool.close()
            pool.join()

        for feature_dict in (r for r in result_map if r is not None):
            this_metric = feature_dict[0]
            result_geo = feature_dict[1]
            result_acc = feature_dict[2]
            unitArea = feature_dict[3]
            ref_regions = feature_dict[4]

            metric_list.append(this_metric)

            # Add scores to images
            for i in ref_regions:
                ind = ref_ndx_orig == i
                image_2d_completeness[ind] = result_geo['2D']['completeness']
                image_2d_correctness[ind] = result_geo['2D']['correctness']
                image_2d_jaccard_index[ind] = result_geo['2D']['jaccardIndex']
                image_3d_completeness[ind] = result_geo['3D']['completeness']
                image_3d_correctness[ind] = result_geo['3D']['correctness']
                image_3d_jaccard_index[ind] = result_geo['3D']['jaccardIndex']
                image_hrmse[ind] = result_acc['hrmse']
                image_zrmse[ind] = result_acc['zrmse']

    else:
        for argument in arguments:
            result = multiprocessing_fun(argument[0], argument[1], argument[2], argument[3], argument[4], argument[5],
                                         argument[6], argument[7], argument[8], argument[9], argument[10], argument[11],
                                         argument[12], argument[13], argument[14], argument[15], argument[16],
                                         argument[17], argument[18], argument[19])

            if result is None:
                continue
            else:
                this_metric = result[0]
                result_geo = result[1]
                result_acc = result[2]
                unitArea = result[3]
                ref_regions = result[4]

            metric_list.append(this_metric)

            # Add scores to images
            for i in ref_regions:
                ind = ref_ndx_orig == i
                image_2d_completeness[ind] = result_geo['2D']['completeness']
                image_2d_correctness[ind] = result_geo['2D']['correctness']
                image_2d_jaccard_index[ind] = result_geo['2D']['jaccardIndex']
                image_3d_completeness[ind] = result_geo['3D']['completeness']
                image_3d_correctness[ind] = result_geo['3D']['correctness']
                image_3d_jaccard_index[ind] = result_geo['3D']['jaccardIndex']
                image_hrmse[ind] = result_acc['hrmse']
                image_zrmse[ind] = result_acc['zrmse']

    # Sort metrics by area
    # Calculate bins for area and volume
    num_bins = 10
    area_range = max_area-min_area
    volume_range = max_volume-min_volume
    area_bin_width = np.round(area_range/num_bins)
    volume_bin_width = np.round(volume_range/num_bins)
    area_bins = []
    volume_bins = []
    for i in range(0, num_bins+1):
        area_bins.append(np.floor(min_area) + i * area_bin_width)
        volume_bins.append(np.floor(min_volume) + i*volume_bin_width)

    # Create dicts with bins as keys
    iou_2d_area_bins = dict((el, []) for el in area_bins)
    iou_2d_volume_bins = dict((el, []) for el in volume_bins)
    iou_3d_area_bins = dict((el, []) for el in area_bins)
    iou_3d_volume_bins = dict((el, []) for el in volume_bins)

    # Create dicts with areas as keys
    iou_2d_area = {}
    iou_2d_volume = {}
    iou_3d_area = {}
    iou_3d_volume = {}

    # Keys are area/volume, Values are IOU
    for current_metric in metric_list:
        current_area = current_metric['threshold_geometry']['area']['test_area']*unitArea  # Convert area to meters^2
        current_volume = current_metric['threshold_geometry']['volume']['test_volume']

        # Get IOUS by area/volume
        iou_2d_area.update({current_area: current_metric['threshold_geometry']['2D']['jaccardIndex']})
        iou_2d_volume.update({current_volume: current_metric['threshold_geometry']['2D']['jaccardIndex']})
        iou_3d_area.update({current_area: current_metric['threshold_geometry']['3D']['jaccardIndex']})
        iou_3d_volume.update({current_volume: current_metric['threshold_geometry']['3D']['jaccardIndex']})

        # Create bins
        for area_bin_edge in area_bins:
            if current_area <= area_bin_edge:
                iou_2d_area_bins[area_bin_edge].append(current_metric['threshold_geometry']['2D']['jaccardIndex'])
                iou_3d_area_bins[area_bin_edge].append(current_metric['threshold_geometry']['3D']['jaccardIndex'])
                break
        for volume_bin_edge in volume_bins:
            if current_volume <= volume_bin_edge:
                iou_2d_volume_bins[volume_bin_edge].append(current_metric['threshold_geometry']['2D']['jaccardIndex'])
                iou_3d_volume_bins[volume_bin_edge].append(current_metric['threshold_geometry']['3D']['jaccardIndex'])
                break

    # Average IOUs in bins
    for current_bin in area_bins:
        iou_2d_area_bins[current_bin] = np.mean(iou_2d_area_bins[current_bin])
        iou_3d_area_bins[current_bin] = np.mean(iou_3d_area_bins[current_bin])
    for current_bin in volume_bins:
        iou_2d_volume_bins[current_bin] = np.mean(iou_2d_volume_bins[current_bin])
        iou_3d_volume_bins[current_bin] = np.mean(iou_3d_volume_bins[current_bin])

    # plot
    if PLOTS_ENABLE:
        print('Input plots...')

        # Save instance level stoplight charts
        plot.make_instance_stoplight_charts(metrics_container_no_merge.stoplight_chart,
                                            saveName=PLOTS_SAVE_PREFIX+"instanceStoplightNoMerge")

        # IOU Histograms
        plot.make_iou_histogram(iou_2d_area_bins, 'Area (m^2)',
                                '2D Mean IOUs by Area', 373, saveName=PLOTS_SAVE_PREFIX +"obj2dIOUbyAreaHistogram")
        plot.make_iou_histogram(iou_2d_volume_bins, 'Volume (m^3)',
                                '2D Mean IOUs by Volume', 374, saveName=PLOTS_SAVE_PREFIX +"obj2dIOUbyVolumeHistogram")
        plot.make_iou_histogram(iou_3d_area_bins, 'Area (m^2)',
                                '3D Mean IOUs by Area', 375, saveName=PLOTS_SAVE_PREFIX +"obj3dIOUbyAreaHistogram")
        plot.make_iou_histogram(iou_3d_volume_bins, 'Volume (m^3)',
                                '3D Mean IOUs by Volume', 376, saveName=PLOTS_SAVE_PREFIX +"obj3dIOUbyVolumeHistogram")
        # Scatter plots
        plot.make_iou_scatter(iou_2d_area, 'Area (m^2)',
                                '2D IOUs by Area', 373, saveName=PLOTS_SAVE_PREFIX +"obj2dIOUbyAreaScatter")
        plot.make_iou_scatter(iou_2d_volume, 'Volume (m^3)',
                                '2D IOUs by Volume', 374, saveName=PLOTS_SAVE_PREFIX +"obj2dIOUbyVolumeScatter")
        plot.make_iou_scatter(iou_3d_area, 'Area (m^2)',
                                '3D IOUs by Area', 375, saveName=PLOTS_SAVE_PREFIX +"obj3dIOUbyAreaScatter")
        plot.make_iou_scatter(iou_3d_volume, 'Volume (m^3)',
                                '3D IOUs by Volume', 376, saveName=PLOTS_SAVE_PREFIX +"obj3dIOUbyVolumeScatter")

        plot.make(image_2d_completeness, 'Objectwise 2D Completeness',
                  351, saveName=PLOTS_SAVE_PREFIX + "obj2dCompleteness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_2d_correctness, 'Objectwise 2D Correctness',
                  352, saveName=PLOTS_SAVE_PREFIX + "obj2dCorrectness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_2d_jaccard_index, 'Objectwise 2D Jaccard Index',
                  353, saveName=PLOTS_SAVE_PREFIX + "obj2dJaccardIndex", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_completeness, 'Objectwise 3D Completeness',
                  361, saveName=PLOTS_SAVE_PREFIX + "obj3dCompleteness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_correctness, 'Objectwise 3D Correctness',
                  362, saveName=PLOTS_SAVE_PREFIX + "obj3dCorrectness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_jaccard_index, 'Objectwise 3D Jaccard Index',
                  363, saveName=PLOTS_SAVE_PREFIX + "obj3dJaccardIndex", colorbar=True, badValue=-1, vmin=0, vmax=1)

        plot.make(image_hrmse, 'Objectwise HRMSE',
                  371, saveName=PLOTS_SAVE_PREFIX+"objHRMSE", colorbar=True, badValue=-1, vmin=0, vmax=2)
        plot.make(image_zrmse, 'Objectwise ZRMSE',
                  372, saveName=PLOTS_SAVE_PREFIX+"objZRMSE", colorbar=True, badValue=-1,  vmin=0, vmax=1)
        plot.make_obj_error_map(error_map=image_hrmse, ref=refMask, badValue=-1,
                                saveName=PLOTS_SAVE_PREFIX + "HRMSE_Image_Only")
        plot.make_obj_error_map(error_map=image_zrmse, ref=refMask, badValue=-1,
                                saveName=PLOTS_SAVE_PREFIX + "ZRMSE_Image_Only")
    # Make per metric reporting structure
    num_objs = len(metric_list)
    summary = {}
    summary['counts'] = {}
    summary['counts']['ref'] = {
        'total':  len(ref_use_counter),
        'used': np.sum(ref_use_counter >= 1).astype(float),
        'unused': np.sum(ref_use_counter == 0).astype(float)
    }

    # Track number of times test objs got reused
    key, val = np.unique(test_use_counter, return_counts=True)

    summary['counts']['test'] = {
        'total':  len(test_use_counter),
        'used': np.sum(test_use_counter >= 1).astype(float),
        'unused': np.sum(test_use_counter == 0).astype(float),
        'counts': {
                    'key':  key.tolist(),
                    'value': val.tolist()
        }
    }

    summary['threshold_geometry'] = {}
    summary['threshold_geometry']['2D'] = {}
    summary['threshold_geometry']['2D']['correctness'] = {}
    summary['threshold_geometry']['2D']['completeness'] = {}
    summary['threshold_geometry']['2D']['jaccardIndex'] = {}
    summary['threshold_geometry']['3D'] = {}
    summary['threshold_geometry']['3D']['correctness'] = {}
    summary['threshold_geometry']['3D']['completeness'] = {}
    summary['threshold_geometry']['3D']['jaccardIndex'] = {}
    summary['relative_accuracy'] = {}
    summary['relative_accuracy']['hrmse'] = {}
    summary['relative_accuracy']['zrmse'] = {}

    summary['threshold_geometry']['2D']['correctness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['2D']['completeness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['2D']['jaccardIndex']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['correctness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['completeness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['jaccardIndex']['values'] = np.zeros(num_objs)
    summary['relative_accuracy']['zrmse']['values'] = np.zeros(num_objs)
    summary['relative_accuracy']['hrmse']['values'] = np.zeros(num_objs)

    ctr = 0
    for m in metric_list:
        summary['threshold_geometry']['2D']['correctness']['values'][ctr] = m['threshold_geometry']['2D']['correctness']
        summary['threshold_geometry']['2D']['completeness']['values'][ctr] = m['threshold_geometry']['2D']['completeness']
        summary['threshold_geometry']['2D']['jaccardIndex']['values'][ctr] = m['threshold_geometry']['2D']['jaccardIndex']

        summary['threshold_geometry']['3D']['correctness']['values'][ctr] = m['threshold_geometry']['3D']['correctness']
        summary['threshold_geometry']['3D']['completeness']['values'][ctr] = m['threshold_geometry']['3D']['completeness']
        summary['threshold_geometry']['3D']['jaccardIndex']['values'][ctr] = m['threshold_geometry']['3D']['jaccardIndex']

        summary['relative_accuracy']['zrmse']['values'][ctr] = m['relative_accuracy']['zrmse']
        summary['relative_accuracy']['hrmse']['values'][ctr] = m['relative_accuracy']['hrmse']

        ctr += 1

    # Compute Summaries
    summary['threshold_geometry']['2D']['correctness'] = metric_stats(summary['threshold_geometry']['2D']['correctness']['values'])
    summary['threshold_geometry']['2D']['completeness'] = metric_stats(summary['threshold_geometry']['2D']['completeness']['values'])
    summary['threshold_geometry']['2D']['jaccardIndex'] = metric_stats(summary['threshold_geometry']['2D']['jaccardIndex']['values'])
    summary['threshold_geometry']['3D']['correctness'] = metric_stats(summary['threshold_geometry']['3D']['correctness']['values'])
    summary['threshold_geometry']['3D']['completeness'] = metric_stats(summary['threshold_geometry']['3D']['completeness']['values'])
    summary['threshold_geometry']['3D']['jaccardIndex'] = metric_stats(summary['threshold_geometry']['3D']['jaccardIndex']['values'])
    summary['relative_accuracy']['zrmse'] = metric_stats(summary['relative_accuracy']['zrmse']['values'])
    summary['relative_accuracy']['hrmse'] = metric_stats(summary['relative_accuracy']['hrmse']['values'])

    # Make summary of metrics
    results = {
        'summary': summary,
        'objects': metric_list,
        'instance_f1': no_merge_f1,
        'instance_f1_merge_fp': None,
        'instance_f1_merge_fn': None,
        'num_buildings_gt': num_buildings_truth,
        'num_buildings_perf': num_buildings_performer,
        'metrics_container_no_merge': metrics_container_no_merge,
        'metrics_container_merge_fp': np.nan,
        'metrics_container_merge_fn': None
    }

    return results, test_ndx, ref_ndx
