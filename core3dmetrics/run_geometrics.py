#
# Run all CORE3D metrics and report results.
#

import os
import errno
import sys
import shutil
import gdalconst
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
try:
    import core3dmetrics.geometrics as geo
except:
    import geometrics as geo


# PRIMARY FUNCTION: RUN_GEOMETRICS
def run_geometrics(config_file, ref_path=None, test_path=None, output_path=None,
                   align=True, allow_test_ignore=False, save_aligned=False, save_plots=None):

    # check inputs
    if not os.path.isfile(config_file):
        raise IOError("Configuration file does not exist")

    if output_path is not None and not os.path.isdir(output_path):
        raise IOError('"output_path" not a valid folder <{}>'.format(output_path))

    # parse configuration
    config_path = os.path.dirname(config_file)

    config = geo.parse_config(config_file,
                              refpath=(ref_path or config_path),
                              testpath=(test_path or config_path))

    # Get test model information from configuration file.
    test_dsm_filename = config['INPUT.TEST']['DSMFilename']
    test_dtm_filename = config['INPUT.TEST'].get('DTMFilename', None)
    test_cls_filename = config['INPUT.TEST']['CLSFilename']
    test_conf_filename = config['INPUT.TEST'].get('CONFFilename', None)
    test_mtl_filename = config['INPUT.TEST'].get('MTLFilename', None)

    # Get reference model information from configuration file.
    ref_dsm_filename = config['INPUT.REF']['DSMFilename']
    ref_dtm_filename = config['INPUT.REF']['DTMFilename']
    ref_cls_filename = config['INPUT.REF']['CLSFilename']
    ref_mtl_filename = config['INPUT.REF'].get('MTLFilename', None)

    # Get material label names and list of material labels to ignore in evaluation.
    material_names = config['MATERIALS.REF']['MaterialNames']
    material_indices_to_ignore = config['MATERIALS.REF']['MaterialIndicesToIgnore']

    # Get image pair files
    performer_pair_file = config['INPUT.TEST'].get('ImagePairFilename', None)
    performer_pair_data_file = config['INPUT.TEST'].get('ImagePairDataFilename', None)
    performer_files_chosen_file = config['INPUT.TEST'].get('FilesChosenFilename', None)
    
    # Get plot settings from configuration file
    PLOTS_SHOW = config['PLOTS']['ShowPlots']
    PLOTS_SAVE = config['PLOTS']['SavePlots']
    if save_plots is not None:  # Commandline line argument overrided config file setting
        PLOTS_SAVE = save_plots
    PLOTS_ENABLE = PLOTS_SHOW or PLOTS_SAVE

    # default output path
    if output_path is None:
        output_path = os.path.dirname(test_dsm_filename)

    if align:
        align = config['OPTIONS']['AlignModel']
    save_aligned = config['OPTIONS']['SaveAligned'] | save_aligned

    # Determine multiprocessing usage
    use_multiprocessing = config['OPTIONS']['UseMultiprocessing']

    # Configure plotting
    basename = os.path.basename(test_dsm_filename)
    if PLOTS_ENABLE:
        plot = geo.plot(saveDir=output_path, autoSave=PLOTS_SAVE, savePrefix=basename + '_', badColor='black', showPlots=PLOTS_SHOW, dpi=900)
    else:
        plot = None

    # copy testDSM to the output path
    # this is a workaround for the "align3d" function with currently always
    # saves new files to the same path as the testDSM
    src = test_dsm_filename
    dst = os.path.join(output_path, os.path.basename(src))
    if not os.path.isfile(dst): shutil.copyfile(src, dst)
    test_dsm_filename_copy = dst

    # Register test model to ground truth reference model.
    if not align:
        print('\nSKIPPING REGISTRATION')
        xyz_offset = (0.0, 0.0, 0.0)
    else:
        print('\n=====REGISTRATION====='); sys.stdout.flush()
        try:
            align3d_path = config['REGEXEPATH']['Align3DPath']
        except:
            align3d_path = None
        xyz_offset = geo.align3d(ref_dsm_filename, test_dsm_filename_copy, exec_path=align3d_path)
        print(xyz_offset)
        #xyz_offset = geo.align3d_python(ref_dsm_filename, test_dsm_filename_copy)

    # Explicitly assign a no data value to warped images to track filled pixels
    no_data_value = -9999

    # Read reference model files.
    print("\nReading reference model files...")
    ref_cls, tform = geo.imageLoad(ref_cls_filename)
    ref_dsm = geo.imageWarp(ref_dsm_filename, ref_cls_filename, noDataValue=no_data_value)
    ref_dtm = geo.imageWarp(ref_dtm_filename, ref_cls_filename, noDataValue=no_data_value)

    # Validate shape of reference files
    if ref_cls.shape != ref_dsm.shape or ref_cls.shape != ref_dtm.shape:
        print("Need to rescale")

    if ref_mtl_filename:
        ref_mtl = geo.imageWarp(ref_mtl_filename, ref_cls_filename, interp_method=gdalconst.GRA_NearestNeighbour).astype(np.uint8)
        if save_aligned:
            geo.arrayToGeotiff(ref_mtl, os.path.join(output_path, basename + '_ref_mtl_reg_out'), ref_cls_filename, no_data_value)
    else:
        ref_mtl = None
        print('NO REFERENCE MTL')

    # Read test model files and apply XYZ offsets.
    print("\nReading test model files...")
    test_cls = geo.imageWarp(test_cls_filename, ref_cls_filename, xyz_offset, gdalconst.GRA_NearestNeighbour)
    test_dsm = geo.imageWarp(test_dsm_filename, ref_cls_filename, xyz_offset, noDataValue=no_data_value)

    if test_dtm_filename:
        test_dtm = geo.imageWarp(test_dtm_filename, ref_cls_filename, xyz_offset, noDataValue=no_data_value)
        if save_aligned:
            geo.arrayToGeotiff(test_dtm, os.path.join(output_path, basename + '_test_dtm_reg_out'), ref_cls_filename,
                               no_data_value)
    else:
        print('NO TEST DTM: defaults to reference DTM')
        test_dtm = ref_dtm

    if test_conf_filename:
        test_conf = geo.imageWarp(test_conf_filename,  ref_cls_filename, xyz_offset, noDataValue=no_data_value)
        conf_viz_path = Path(str(Path(test_conf_filename).parent.absolute()),
                             Path(test_conf_filename).stem + '_VIZ.tif')
        test_conf_viz = geo.imageWarpRGB(str(conf_viz_path.absolute()), ref_cls_filename, xyz_offset)
        geo.arrayToGeotiffRGB(test_conf_viz, os.path.join(output_path, 'CONF_VIZ_aligned'), ref_cls_filename,
                              no_data_value)

        geo.arrayToGeotiff(test_conf, os.path.join(output_path, 'CONF_aligned'), ref_cls_filename,
                           no_data_value)
    else:
        test_conf = None
        print("NO TEST CONF")

    if save_aligned:
        geo.arrayToGeotiff(test_cls, os.path.join(output_path, basename + '_test_cls_reg_out'), ref_cls_filename,
                           no_data_value)
        geo.arrayToGeotiff(test_dsm, os.path.join(output_path, basename + '_test_dsm_reg_out'), ref_cls_filename,
                           no_data_value)
        geo.arrayToGeotiff(ref_cls, os.path.join(output_path, basename + '_ref_cls_reg_out'), ref_cls_filename,
                           no_data_value)
        geo.arrayToGeotiff(ref_dsm, os.path.join(output_path, basename + '_ref_dsm_reg_out'), ref_cls_filename,
                           no_data_value)
        geo.arrayToGeotiff(ref_dtm, os.path.join(output_path, basename + '_ref_dtm_reg_out'), ref_cls_filename,
                           no_data_value)

    if test_mtl_filename:
        test_mtl = geo.imageWarp(test_mtl_filename, ref_cls_filename, xyz_offset,
                                 gdalconst.GRA_NearestNeighbour).astype(np.uint8)
        if save_aligned:
            geo.arrayToGeotiff(test_mtl, os.path.join(output_path, basename + '_test_mtl_reg_out'), ref_cls_filename,
                               no_data_value)
    else:
        print('NO TEST MTL')

    print("\n\n")

    # Apply registration offset, only to valid data to allow better tracking of bad data
    print("Applying offset of Z:  %f" % (xyz_offset[2]))
    test_valid_data = (test_dsm != no_data_value)
    if test_dtm_filename:
        test_valid_data &= (test_dtm != no_data_value)

    test_dsm[test_valid_data] = test_dsm[test_valid_data] + xyz_offset[2]
    if test_dtm_filename:
        test_dtm[test_valid_data] = test_dtm[test_valid_data] + xyz_offset[2]

    # Create mask for ignoring points labeled NoData in reference files.
    ref_dsm_no_data_value = no_data_value
    ref_dtm_no_data_value = no_data_value
    ref_cls_no_data_value = geo.getNoDataValue(ref_cls_filename)
    if ref_cls_no_data_value != 65:
        print("WARNING! NODATA TAG IN CLS FILE IS LIKELY INCORRECT. IT SHOULD BE 65.")
        ref_cls_no_data_value = 65
    ignore_mask = np.zeros_like(ref_cls, np.bool)

    # Get reference and test classifications
    ref_cls_match_sets, test_cls_match_sets = geo.getMatchValueSets(config['INPUT.REF']['CLSMatchValue'],
                                                                    config['INPUT.TEST']['CLSMatchValue'],
                                                                    np.unique(ref_cls).tolist(),
                                                                    np.unique(test_cls).tolist())
    # Add ignore mask on boundaries of cls
    # Ignore edges
    ignore_edges = False
    if ignore_edges is True:
        print("Applying ignore mask to edges of buildings...")
        for index, (ref_match_value, test_match_value) in enumerate(zip(ref_cls_match_sets, test_cls_match_sets)):
            import scipy.ndimage as ndimage
            ref_mask = np.zeros_like(ref_cls, np.bool)
            for v in ref_match_value:
                ref_mask[ref_cls == v] = True
            strel = ndimage.generate_binary_structure(2, 2)
            dilated_cls = ndimage.binary_dilation(ref_mask, structure=strel, iterations=3)
            eroded_cls = ndimage.binary_erosion(ref_mask, structure=strel, iterations=3)
            dilation_mask = np.bitwise_xor(ref_mask, dilated_cls)
            erosion_mask = np.bitwise_xor(ref_mask, eroded_cls)
            ref_cls[dilation_mask == True] = ref_cls_no_data_value
            ref_cls[erosion_mask == True] = ref_cls_no_data_value
        print("Finished applying ignore mask to edges of buildings...")

    # Create ignore mask
    if ref_dsm_no_data_value is not None:
        ignore_mask[ref_dsm == ref_dsm_no_data_value] = True
    if ref_dtm_no_data_value is not None:
        ignore_mask[ref_dtm == ref_dtm_no_data_value] = True
    if ref_cls_no_data_value is not None:
        ignore_mask[ref_cls == ref_cls_no_data_value] = True

    # optionally ignore test NoDataValue(s)
    if allow_test_ignore:
        if allow_test_ignore == 1:
            test_cls_no_data_value = geo.getNoDataValue(test_cls_filename)
            if test_cls_no_data_value is not None:
                print('Ignoring test CLS NoDataValue')
                ignore_mask[test_cls == test_cls_no_data_value] = True

        elif allow_test_ignore == 2:
            test_dsm_no_data_value = no_data_value
            test_dtm_no_data_value = no_data_value
            if test_dsm_no_data_value is not None:
                print('Ignoring test DSM NoDataValue')
                ignore_mask[test_dsm == test_dsm_no_data_value] = True
            if test_dtm_filename and test_dtm_no_data_value is not None:
                print('Ignoring test DTM NoDataValue')
                ignore_mask[test_dtm == test_dtm_no_data_value] = True

        else:
            raise IOError('Unrecognized test ignore value={}'.format(allow_test_ignore))

        print("")

    # sanity check
    if np.all(ignore_mask):
        raise ValueError('All pixels are ignored')

    ##### COMMENT HERE FOR TESTING METICS IMAGES #####
    # report "data voids"
    num_data_voids = np.sum(ignore_mask > 0)
    print('Number of data voids in ignore mask = ', num_data_voids)

    # If quantizing to voxels, then match vertical spacing to horizontal spacing.
    QUANTIZE = config['OPTIONS']['QuantizeHeight']
    if QUANTIZE:
        unit_hgt = geo.getUnitHeight(tform)
        ref_dsm = np.round(ref_dsm / unit_hgt) * unit_hgt
        ref_dtm = np.round(ref_dtm / unit_hgt) * unit_hgt
        test_dsm = np.round(test_dsm / unit_hgt) * unit_hgt
        test_dtm = np.round(test_dtm / unit_hgt) * unit_hgt
        no_data_value = np.round(no_data_value / unit_hgt) * unit_hgt

    if PLOTS_ENABLE:
        # Make image pair plots
        plot.make_image_pair_plots(performer_pair_data_file, performer_pair_file, performer_files_chosen_file, 201,
                                   saveName="image_pair_plot")
        # Reference models can include data voids, so ignore invalid data on display
        plot.make(ref_dsm, 'Reference DSM', 111, colorbar=True, saveName="input_refDSM", badValue=no_data_value)
        plot.make(ref_dtm, 'Reference DTM', 112, colorbar=True, saveName="input_refDTM", badValue=no_data_value)
        plot.make(ref_cls, 'Reference Classification', 113,  colorbar=True, saveName="input_refClass")

        # Test models shouldn't have any invalid data
        # so display the invalid values to highlight them,
        # unlike with the refSDM/refDTM
        plot.make(test_dsm, 'Test DSM', 151, colorbar=True, saveName="input_testDSM")
        plot.make(test_dtm, 'Test DTM', 152, colorbar=True, saveName="input_testDTM")
        plot.make(test_cls, 'Test Classification', 153, colorbar=True, saveName="input_testClass")

        plot.make(ignore_mask, 'Ignore Mask', 181, saveName="input_ignoreMask")

        # material maps
        if ref_mtl_filename and test_mtl_filename:
            plot.make(ref_mtl, 'Reference Materials', 191, colorbar=True, saveName="input_refMTL", vmin=0, vmax=13)
            plot.make(test_mtl, 'Test Materials', 192, colorbar=True, saveName="input_testMTL", vmin=0, vmax=13)

    # Run the threshold geometry metrics and report results.
    metrics = dict()

    # Run threshold geometry and relative accuracy
    threshold_geometry_results = []
    relative_accuracy_results = []
    objectwise_results = []

    if PLOTS_ENABLE:
        # Update plot prefix include counter to be unique for each set of CLS value evaluated
        original_save_prefix = plot.savePrefix

    # Loop through sets of CLS match values
    for index, (ref_match_value,test_match_value) in enumerate(zip(ref_cls_match_sets, test_cls_match_sets)):
        print("Evaluating CLS values")
        print("  Reference match values: " + str(ref_match_value))
        print("  Test match values: " + str(test_match_value))

        # object masks based on CLSMatchValue(s)
        ref_mask = np.zeros_like(ref_cls, np.bool)
        for v in ref_match_value:
            ref_mask[ref_cls == v] = True

        test_mask = np.zeros_like(test_cls, np.bool)
        if len(test_match_value):
            for v in test_match_value:
                test_mask[test_cls == v] = True

        if PLOTS_ENABLE:
            plot.savePrefix = original_save_prefix + "%03d" % index + "_"
            plot.make(test_mask.astype(np.int), 'Test Evaluation Mask', 154, saveName="input_testMask")
            plot.make(ref_mask.astype(np.int), 'Reference Evaluation Mask', 114, saveName="input_refMask")

        if config['OBJECTWISE']['Enable']:
            print("\nRunning objectwise metrics...")
            merge_radius = config['OBJECTWISE']['MergeRadius']
            [result, test_ndx, ref_ndx] = geo.run_objectwise_metrics(ref_dsm, ref_dtm, ref_mask, test_dsm, test_dtm,
                                                                     test_mask, tform, ignore_mask, merge_radius,
                                                                     plot=plot, geotiff_filename=ref_dsm_filename,
                                                                     use_multiprocessing=use_multiprocessing)

            # Get UTM coordinates from pixel coordinates in building centroids
            print("Creating KML and CSVs...")
            import gdal, osr, simplekml, csv
            kml = simplekml.Kml()
            ds = gdal.Open(ref_dsm_filename)
            # get CRS from dataset
            crs = osr.SpatialReference()
            crs.ImportFromWkt(ds.GetProjectionRef())
            # create lat/long crs with WGS84 datum
            crsGeo = osr.SpatialReference()
            crsGeo.ImportFromEPSG(4326)  # 4326 is the EPSG id of lat/long crs
            t = osr.CoordinateTransformation(crs, crsGeo)
            # Use CORE3D objectwise
            current_class = test_match_value[0]
            with open(Path(output_path, "objectwise_numbers_class_" + str(current_class) + ".csv"), mode='w') as \
                    objectwise_csv:
                objectwise_writer = csv.writer(objectwise_csv, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                objectwise_writer.writerow(
                    ['Index', 'iou_2d', 'iou_3d', 'hrmse', 'zrmse', 'x_coord', 'y_coord', 'geo_x_coord',
                     'geo_y_coord', 'long', 'lat'])
                for current_object in result['objects']:
                    test_index = current_object['test_objects'][0]
                    iou_2d = current_object['threshold_geometry']['2D']['jaccardIndex']
                    iou_3d = current_object['threshold_geometry']['3D']['jaccardIndex']
                    hrmse = current_object['relative_accuracy']['hrmse']
                    zrmse = current_object['relative_accuracy']['zrmse']
                    x_coords, y_coords = np.where(test_ndx == test_index)
                    x_coord = np.average(x_coords)
                    y_coord = np.average(y_coords)
                    geo_x_coord = tform[0] + y_coord * tform[1] + x_coord * tform[2]
                    geo_y_coord = tform[3] + y_coord * tform[4] + x_coord * tform[5]
                    (lat, long, z) = t.TransformPoint(geo_x_coord, geo_y_coord)
                    objectwise_writer.writerow([test_index, iou_2d, iou_3d, hrmse, zrmse, x_coord, y_coord,
                                                geo_x_coord, geo_y_coord, long, lat])
                    pnt = kml.newpoint(name="Building Index: " + str(test_index),
                                       description="2D IOU: " + str(iou_2d) + ' 3D IOU: ' + str(iou_3d) + ' HRMSE: '
                                                   + str(hrmse) + ' ZRMSE: ' + str(zrmse),
                                       coords=[(lat, long)])
                kml.save(Path(output_path, "objectwise_ious_class_" + str(current_class) + ".kml"))

            # Use FFDA objectwise
            with open(Path(output_path, "objectwise_numbers_no_morphology_class_" + str(current_class) + ".csv"),
                      mode='w') as objectwise_csv:
                objectwise_writer = csv.writer(objectwise_csv, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                objectwise_writer.writerow(['iou', 'x_coord', 'y_coord', 'geo_x_coord',
                                            'geo_y_coord', 'long', 'lat'])
                for i in result['metrics_container_no_merge'].iou_per_gt_building.keys():
                    iou = result['metrics_container_no_merge'].iou_per_gt_building[i][0]
                    x_coord = result['metrics_container_no_merge'].iou_per_gt_building[i][1][0]
                    y_coord = result['metrics_container_no_merge'].iou_per_gt_building[i][1][1]
                    geo_x_coord = tform[0] + y_coord * tform[1] + x_coord * tform[2]
                    geo_y_coord = tform[3] + y_coord * tform[4] + x_coord * tform[5]
                    (lat, long, z) = t.TransformPoint(geo_x_coord, geo_y_coord)
                    objectwise_writer.writerow([iou, x_coord, y_coord, geo_x_coord, geo_y_coord, long, lat])
                    pnt = kml.newpoint(name="Building Index: " + str(i), description=str(iou), coords=[(lat, long)])
            kml.save(Path(output_path, "objectwise_ious_no_morphology_class_" + str(current_class) + ".kml"))
            # Result
            if ref_match_value == test_match_value:
                result['CLSValue'] = ref_match_value
            else:
                result['CLSValue'] = {'Ref': ref_match_value, "Test": test_match_value}
            # Delete non-json dumpable metrics
            del result['metrics_container_no_merge'], result['metrics_container_merge_fp'], result[
                'metrics_container_merge_fn']
            objectwise_results.append(result)

            # Save index files to compute objectwise metrics
            obj_save_prefix = basename + "_%03d" % index + "_"
            geo.arrayToGeotiff(test_ndx, os.path.join(output_path, obj_save_prefix + '_test_ndx_objs'),
                               ref_cls_filename, no_data_value)
            geo.arrayToGeotiff(ref_ndx, os.path.join(output_path, obj_save_prefix + '_ref_ndx_objs'),
                               ref_cls_filename,
                               no_data_value)

        # Evaluate threshold geometry metrics using refDTM as the testDTM to mitigate effects of terrain modeling
        # uncertainty
        result, _, stoplight_fn, errhgt_fn = geo.run_threshold_geometry_metrics(ref_dsm, ref_dtm, ref_mask, test_dsm, test_dtm, test_mask, tform,
                                                    ignore_mask, testCONF=test_conf, plot=plot)
        if ref_match_value == test_match_value:
            result['CLSValue'] = ref_match_value
        else:
            result['CLSValue'] = {'Ref': ref_match_value, "Test": test_match_value}
        threshold_geometry_results.append(result)

        # Run the relative accuracy metrics and report results.
        # Skip relative accuracy is all of testMask or refMask is assigned as "object"
        if not ((ref_mask.size == np.count_nonzero(ref_mask)) or (test_mask.size == np.count_nonzero(test_mask))) and len(test_match_value) != 0:
            try:
                result = geo.run_relative_accuracy_metrics(ref_dsm, test_dsm, ref_mask, test_mask, ignore_mask,
                                                           geo.getUnitWidth(tform), plot=plot)
                if ref_match_value == test_match_value:
                    result['CLSValue'] = ref_match_value
                else:
                    result['CLSValue'] = {'Ref': ref_match_value, "Test": test_match_value}
                relative_accuracy_results.append(result)
            except Exception as e:
                print(str(e))

    if PLOTS_ENABLE:
        # Reset plot prefix
        plot.savePrefix = original_save_prefix

    metrics['threshold_geometry'] = threshold_geometry_results
    metrics['relative_accuracy'] = relative_accuracy_results
    metrics['objectwise'] = objectwise_results

    if align:
        metrics['registration_offset'] = xyz_offset
        metrics['geolocation_error'] = np.linalg.norm(xyz_offset)

    # Run the terrain model metrics and report results.
    if test_dtm_filename:
        dtm_z_threshold = config['OPTIONS'].get('TerrainZErrorThreshold', 1)

        # Make reference mask for terrain evaluation that identified elevated object where underlying terrain estimate
        # is expected to be inaccurate
        dtm_cls_ignore_values = config['INPUT.REF'].get('TerrainCLSIgnoreValues', [6, 17]) # Default to building and bridge deck
        dtm_cls_ignore_values = geo.validateMatchValues(dtm_cls_ignore_values,np.unique(ref_cls).tolist())
        ref_mask_terrain_acc = np.zeros_like(ref_cls, np.bool)
        for v in dtm_cls_ignore_values:
            ref_mask_terrain_acc[ref_cls == v] = True

        metrics['terrain_accuracy'] = geo.run_terrain_accuracy_metrics(ref_dtm, test_dtm, ref_mask_terrain_acc,
                                                                       dtm_z_threshold, plot=plot)
    else:
        print('WARNING: No test DTM file, skipping terrain accuracy metrics')

    # Run the threshold material metrics and report results.
    if test_mtl_filename and ref_mtl:
        metrics['threshold_materials'] = geo.run_material_metrics(ref_ndx, ref_mtl, test_mtl, material_names,
                                                                  material_indices_to_ignore, plot=plot)
    else:
        print('WARNING: No test MTL file or no reference material, skipping material metrics')

    fileout = os.path.join(output_path, os.path.basename(config_file) + "_metrics.json")
    with open(fileout, 'w') as fid:
        json.dump(metrics, fid, indent=2)
    print(json.dumps(metrics, indent=2))
    print("Metrics report: " + fileout)

    #  If displaying figures, wait for user before exiting
    if PLOTS_SHOW:
            input("Press Enter to continue...")

    # Write final metrics out
    output_folder = os.path.join(output_path, "metrics_final")
    try:
        os.mkdir(output_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            print("Can't create directory, please check permissions...")
            raise

    # Run Roof slope metrics
    # Roof Slope Metrics
    try:
        from core3dmetrics.geometrics.ang import calculate_metrics as calculate_roof_metrics
    except:
        from ang import calculate_metrics as calculate_roof_metrics

    IOUC, IOUZ, IOUAGL, IOUMZ, orderRMS = calculate_roof_metrics(ref_dsm, ref_dtm, ref_cls, test_dsm, test_dtm,
                           test_cls, tform, kernel_radius=3, output_path=output_path)
    files = [str(Path(output_path, filename).absolute()) for filename in os.listdir(output_path) if
             filename.startswith("Roof")]

    # Save all of myrons outputs here
    metrics_formatted = {}
    metrics_formatted["2D"] = {}
    metrics_formatted["2D"]["Precision"] = metrics["threshold_geometry"][0]['2D']['precision']
    metrics_formatted["2D"]["Recall"] = metrics["threshold_geometry"][0]['2D']['recall']
    metrics_formatted["2D"]["IOU"] = metrics["threshold_geometry"][0]['2D']['jaccardIndex']
    metrics_formatted["3D"] = {}
    metrics_formatted["3D"]["Precision"] = metrics["threshold_geometry"][0]['3D']['precision']
    metrics_formatted["3D"]["Recall"] = metrics["threshold_geometry"][0]['3D']['recall']
    metrics_formatted["3D"]["IOU"] = metrics["threshold_geometry"][0]['3D']['jaccardIndex']
    metrics_formatted["ZRMS"] = metrics['relative_accuracy'][0]['zrmse']
    metrics_formatted["HRMS"] = metrics['relative_accuracy'][0]['hrmse']
    metrics_formatted["Slope RMS"] = orderRMS
    metrics_formatted["DTM RMS"] = metrics['terrain_accuracy']['zrmse']
    metrics_formatted["DTM Completeness"] = metrics['terrain_accuracy']['completeness']
    metrics_formatted["Z IOU"] = IOUZ
    metrics_formatted["AGL IOU"] = IOUAGL
    metrics_formatted["MODEL IOU"] = IOUMZ
    metrics_formatted["X Offset"] = xyz_offset[0]
    metrics_formatted["Y Offset"] = xyz_offset[1]
    metrics_formatted["Z Offset"] = xyz_offset[2]
    metrics_formatted["P Value"] = metrics['threshold_geometry'][0]['pearson']

    fileout = os.path.join(output_folder, "metrics.json")
    with open(fileout, 'w') as fid:
        json.dump(metrics_formatted, fid, indent=2)
    print(json.dumps(metrics_formatted, indent=2))

    # metrics.png
    if PLOTS_ENABLE:
        cls_iou_fn = [filename for filename in files if filename.endswith("CLS_IOU.tif")][0]
        cls_z_iou_fn = [filename for filename in files if filename.endswith("CLS_Z_IOU.tif")][0]
        cls_z_slope_fn = [filename for filename in files if filename.endswith("CLS_Z_SLOPE_IOU.tif")][0]
        if test_conf_filename:
            plot.make_final_metrics_images(stoplight_fn, errhgt_fn, Path(os.path.join(output_path, 'CONF_VIZ_aligned.tif')),
                                           cls_iou_fn, cls_z_iou_fn, cls_z_slope_fn, ref_cls, output_folder)

    # inputs.png
        plot.make_final_input_images_grayscale(ref_cls, ref_dsm, ref_dtm, test_cls,
                                                test_dsm, test_dtm, output_folder)
    # textured.png
    if config['BLENDER.TEST']['OBJDirectoryFilename']:
        try:
            from CORE3D_Perspective_Imagery import generate_blender_images
            objpath = config['BLENDER.TEST']['OBJDirectoryFilename']
            gsd = config['BLENDER.TEST']['GSD']
            Zup = config['BLENDER.TEST']['+Z']
            N = config['BLENDER.TEST']['OrbitalLocations']
            e = config['BLENDER.TEST']['ElevationAngle']
            f = config['BLENDER.TEST']['FocalLength']
            r = config['BLENDER.TEST']['RadialDistance']
            output_location = generate_blender_images(objpath, gsd, Zup, N, e, f, r, output_path)
            files = [str(Path(output_path, filename).absolute()) for filename in os.listdir(output_path) if
                     filename.startswith("persp_image")]
            files.append(files[0])
            # Make metrics image
            plot.make_final_input_images_rgb(files, output_folder)
            print("Done")
        except:
            print("Could not render Blender images...")
    else:
        pass



# command line function
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # parse inputs
    parser = argparse.ArgumentParser(description='core3dmetrics entry point', prog='core3dmetrics')

    parser.add_argument('-c', '--config', dest='config',
                        help='Configuration file', required=True, metavar='')
    parser.add_argument('-r', '--reference', dest='refpath',
                        help='Reference data folder', required=False, metavar='')
    parser.add_argument('-t', '--test', dest='testpath',
                        help='Test data folder', required=False, metavar='')
    parser.add_argument('-o', '--output', dest='outputpath',
                        help='Output folder', required=False, metavar='')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--align', dest='align', action='store_true', help="Enable alignment (default)")
    group.add_argument('--no-align', dest='align', action='store_false', help="Disable alignment")
    group.set_defaults(align=True)

    # optional argument, enables saving of aligned image to disk
    parser.add_argument('--save-aligned', dest='savealigned', required=False, action='store_true',
                        help="Save aligned images (not enabled by default)")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--save-plots', dest='saveplots', action='store_true', help="Save plots. Overrides config file setting.")
    group.add_argument('--skip-save-plots', dest='saveplots', action='store_false', help="Don't save plots. Overrides config file setting.")
    group.set_defaults(saveplots=None)

    # optional argument
    # note if "--test-ignore" specified without argument, testignore==1
    parser.add_argument('--test-ignore', dest='testignore',
                        help="Ignore test NoDataValue(s) (0=off, 1=ignore CLS, 2=ignore DSM/DTM",
                        required=False, nargs='?', default=0, const=1,
                        choices=range(0,3), type=int, metavar='')

    args, unknown = parser.parse_known_args(args)

    start_time = datetime.now()
    print('RUN_GEOMETRICS input arguments:')
    print(args)

    if unknown:
        print('Unknown input arguments:')
        print(unknown)
        return

    # gather optional arguments
    kwargs = {}
    kwargs['align'] = args.align
    if args.refpath: kwargs['refpath'] = args.refpath
    if args.testpath: kwargs['test_path'] = args.testpath
    if args.outputpath: kwargs['output_path'] = args.outputpath
    if args.testignore: kwargs['allow_test_ignore'] = args.testignore
    if args.savealigned: kwargs['save_aligned'] = args.savealigned
    if args.saveplots is not None: kwargs['save_plots'] = args.saveplots

    # run process
    run_geometrics(config_file=args.config, **kwargs)
    elapsed_time = datetime.now() - start_time
    print("Elapsed time: " + str(elapsed_time))


if __name__ == "__main__":
    main()

