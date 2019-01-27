
import numpy as np

import scipy.ndimage as ndimage



from .metrics_util import getUnitWidth

try:
    import core3dmetrics.geometrics as geo
except:
    import geometrics as geo

def eval_metrcs(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, plot=None, verbose=True):

    # Evaluate threshold geometry metrics using refDTM as the testDTM to mitigate effects of terrain modeling uncertainty
    result_geo = geo.run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, refDTM, testMask, tform, ignoreMask,
                                                plot=plot, verbose=verbose)

    # Run the relative accuracy metrics and report results.
    result_acc = geo.run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask,
                                                   geo.getUnitWidth(tform), plot=plot)


    return result_geo, result_acc


# Compute statistics on a list of values
def metric_stats(val):
    s = dict()
    s['values'] = val.tolist()
    s['mean']   = np.mean(val)
    s['stddev'] = np.std(val)
    s['pctl'] = {}
    s['pctl']['rank']  = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 91, 92, 93, 94, 95, 96, 96, 98, 99, 100]
    s['pctl']['value'] = np.percentile(val, s['pctl']['rank']).tolist()
    return s



def run_objectwise_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, plot=None, verbose=True):

    padding_meters = 2

    # Number of pixels to dilate reference object mask
    padding_pixels = np.round(padding_meters / getUnitWidth(tform))
    strel = ndimage.generate_binary_structure(2, 1)

    # Dilate reference object mask to combine closely spaced objects
    refNdxOrig =  np.copy(refMask)
    refNdx = ndimage.binary_dilation(refNdxOrig, structure=strel,  iterations=padding_pixels.astype(int))

    # Create index regions
    refNdx, num_ref_regions = ndimage.label(refNdx)
    refNdxOrig, num_ref_regions = ndimage.label(refNdxOrig)
    testNdx, num_test_regions = ndimage.label(testMask)

    # Keep track of how many times each region is used
    testUseCounter = np.zeros([num_test_regions ,1])
    refUseCounter  = np.zeros([num_ref_regions, 1])

    metric_list = []

    for loopRegion in range(1,num_ref_regions):

        # Reference region under evaluation
        refObjs = (refNdx == loopRegion) & refMask

        # Find test regions overlapping with ref
        testRegions = np.unique(testNdx[refNdx == loopRegion])

        # Find test regions overlapping with ref
        refRegions = np.unique(refNdxOrig[refNdx == loopRegion])

        # Remove background region, '0'
        if np.any(testRegions == 0):
            testRegions = testRegions.tolist()
            testRegions.remove(0)
            testRegions = np.array(testRegions)

        if np.any(refRegions == 0):
            refRegions = refRegions.tolist()
            refRegions.remove(0)
            refRegions = np.array(refRegions)


        if len(testRegions) == 0:
            continue


        for refRegion in refRegions:
            # Increment counter for ref region used
            refUseCounter[refRegion - 1] = refUseCounter[refRegion - 1] + 1


        # Make mask of overlapping test regions
        testObjs = np.zeros_like(testMask)
        for testRegion in testRegions:
            testObjs = testObjs | (testNdx == testRegion)
            # Increment counter for test region used
            testUseCounter[testRegion-1] = testUseCounter[testRegion-1] + 1


        # TODO:  Not practical as implemented to enable plots. plots is forced to false.
        [result_geo, result_acc] = eval_metrcs(refDSM, refDTM, refObjs, testDSM, testDTM, testObjs, tform, ignoreMask, plot=None, verbose=verbose)

        this_metric = dict()
        this_metric['ref_objects'] = testRegions.tolist()
        this_metric['test_objects'] = refRegions.tolist()
        this_metric['threshold_geometry'] = result_geo
        this_metric['relative_accuracy'] = result_acc

        metric_list.append(this_metric)


    # Make per metric reporting structure
    num_objs = len(metric_list)
    summary = {}
    summary['counts'] = {}
    summary['counts']['ref'] = {
        'total' :  len(refUseCounter),
        'used': np.sum(refUseCounter >= 1).astype(float),
        'unused': np.sum(refUseCounter == 0).astype(float)
    }

    # Track number of times test objs got reused
    key, val = np.unique(testUseCounter, return_counts=True)

    summary['counts']['test'] = {
        'total' :  len(testUseCounter),
        'used': np.sum(testUseCounter >= 1).astype(float),
        'unused': np.sum(testUseCounter == 0).astype(float),
        'counts' : {
                    'key' :  key.tolist(),
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

    summary['threshold_geometry']['2D']['correctness']['values']  = np.zeros(num_objs)
    summary['threshold_geometry']['2D']['completeness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['2D']['jaccardIndex']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['correctness']['values']  = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['completeness']['values'] = np.zeros(num_objs)
    summary['threshold_geometry']['3D']['jaccardIndex']['values'] = np.zeros(num_objs)
    summary['relative_accuracy']['zrmse']['values'] = np.zeros(num_objs)
    summary['relative_accuracy']['hrmse']['values'] = np.zeros(num_objs)

    ctr = 0
    for m in  metric_list:
        summary['threshold_geometry']['2D']['correctness']['values'][ctr]  = m['threshold_geometry']['2D']['correctness']
        summary['threshold_geometry']['2D']['completeness']['values'][ctr] = m['threshold_geometry']['2D']['completeness']
        summary['threshold_geometry']['2D']['jaccardIndex']['values'][ctr] = m['threshold_geometry']['2D']['jaccardIndex']

        summary['threshold_geometry']['3D']['correctness']['values'][ctr]  = m['threshold_geometry']['3D']['correctness']
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
    results  = {
        'summary':summary,
        'objects': metric_list
    }


    return results, testNdx, refNdx