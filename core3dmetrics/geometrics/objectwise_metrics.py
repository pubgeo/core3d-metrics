
import numpy as np

import scipy.ndimage as ndimage



from .metrics_util import getUnitWidth
from .threshold_geometry_metrics import run_threshold_geometry_metrics
from .relative_accuracy_metrics import run_relative_accuracy_metrics


def eval_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, plot=None, verbose=True):

    # Evaluate threshold geometry metrics using refDTM as the testDTM to mitigate effects of terrain modeling
    # uncertainty
    result_geo = run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, refDTM, testMask, tform, ignoreMask,
                                                plot=plot, verbose=verbose)

    # Run the relative accuracy metrics and report results.
    result_acc = run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask,
                                               getUnitWidth(tform), plot=plot)

    return result_geo, result_acc


# Compute statistics on a list of values
def metric_stats(val):
    s = dict()
    s['values'] = val.tolist()
    s['mean'] = np.mean(val)
    s['stddev'] = np.std(val)
    s['pctl'] = {}
    s['pctl']['rank'] = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 91, 92, 93, 94, 95, 96, 96, 98, 99, 100]
    s['pctl']['value'] = np.percentile(val, s['pctl']['rank']).tolist()
    return s


def run_objectwise_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, merge_radius=2, plot=None, verbose=True):

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
    ref_ndx = ndimage.binary_dilation(ref_ndx_orig, structure=strel,  iterations=padding_pixels.astype(int))

    # Create index regions
    ref_ndx, num_ref_regions = ndimage.label(ref_ndx)
    ref_ndx_orig, num_ref_regions = ndimage.label(ref_ndx_orig)
    test_ndx, num_test_regions = ndimage.label(testMask)

    # Keep track of how many times each region is used
    test_use_counter = np.zeros([num_test_regions, 1])
    ref_use_counter = np.zeros([num_ref_regions, 1])

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
    for loop_region in range(1, num_ref_regions+1):

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
            continue

        for refRegion in ref_regions:
            # Increment counter for ref region used
            ref_use_counter[refRegion - 1] = ref_use_counter[refRegion - 1] + 1

        # Make mask of overlapping test regions
        test_objs = np.zeros_like(testMask)
        for test_region in test_regions:
            test_objs = test_objs | (test_ndx == test_region)
            # Increment counter for test region used
            test_use_counter[test_region-1] = test_use_counter[test_region-1] + 1

        # TODO:  Not practical as implemented to enable plots. plots is forced to false.
        [result_geo, result_acc] = eval_metrics(refDSM, refDTM, ref_objs, testDSM, testDTM, test_objs, tform, ignoreMask, plot=None, verbose=verbose)

        this_metric = dict()
        this_metric['ref_objects'] = test_regions.tolist()
        this_metric['test_objects'] = ref_regions.tolist()
        this_metric['threshold_geometry'] = result_geo
        this_metric['relative_accuracy'] = result_acc

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

    # plot
    if PLOTS_ENABLE:
        print('Input plots...')

        plot.make(image_2d_completeness, 'Objectwise 2D Completeness', 351, saveName=PLOTS_SAVE_PREFIX + "obj2dCompleteness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_2d_correctness, 'Objectwise 2D Correctness', 352, saveName=PLOTS_SAVE_PREFIX + "obj2dCorrectness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_2d_jaccard_index, 'Objectwise 2D Jaccard Index', 353, saveName=PLOTS_SAVE_PREFIX + "obj2dJaccardIndex", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_completeness, 'Objectwise 3D Completeness', 361, saveName=PLOTS_SAVE_PREFIX + "obj3dCompleteness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_correctness, 'Objectwise 3D Correctness', 362, saveName=PLOTS_SAVE_PREFIX + "obj3dCorrectness", colorbar=True, badValue=-1, vmin=0, vmax=1)
        plot.make(image_3d_jaccard_index, 'Objectwise 3D Jaccard Index', 363, saveName=PLOTS_SAVE_PREFIX + "obj3dJaccardIndex", colorbar=True, badValue=-1, vmin=0, vmax=1)

        plot.make(image_hrmse, 'Objectwise HRMSE', 371, saveName=PLOTS_SAVE_PREFIX+"objHRMSE", colorbar=True, badValue=-1, vmin=0, vmax=2)
        plot.make(image_zrmse, 'Objectwise ZRMSE', 372, saveName=PLOTS_SAVE_PREFIX+"objZRMSE", colorbar=True, badValue=-1,  vmin=0, vmax=1)

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
        'objects': metric_list
    }

    return results, test_ndx, ref_ndx
