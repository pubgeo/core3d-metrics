import json
import jsonschema
import numpy as np
from pathlib import Path
from MetricContainer import Result
try:
    import core3dmetrics.geometrics as geo
except:
    import geometrics as geo


# BAA Thresholds
class BAAThresholds:

    def __init__(self):
        self.geolocation_error = np.array([2, 1.5, 1.5, 1])*3.5
        self.completeness_2d = np.array([0.8, 0.85, 0.9, 0.95])
        self.correctness_2d = np.array([0.8, 0.85, 0.9, 0.95])
        self.completeness_3d = np.array([0.6, 0.7, 0.8, 0.9])
        self.correctness_3d = np.array([0.6, 0.7, 0.8, 0.9])
        self.material_accuracy = np.array([0.85, 0.90, 0.95, 0.98])
        self.model_build_time = np.array([8, 2, 2, 1])
        self.fscore_2d = (2*self.completeness_2d * self.correctness_2d) / (self.completeness_2d + self.correctness_2d)
        self.fscore_3d = (2*self.completeness_3d * self.correctness_3d) / (self.completeness_3d + self.correctness_3d)
        self.jaccard_index_2d = np.round(self.fscore_2d / (2-self.fscore_2d), decimals=2)
        self.jaccard_index_3d = np.round(self.fscore_3d / (2-self.fscore_3d), decimals=2)


def summarize_metrics(root_dir, teams, aois, ref_path=None, test_path=None):
    # load results
    is_config = True
    all_results = {}
    # Parse results
    for current_team in teams:
        for current_aoi in aois:
            metrics_json_filepath = Path(root_dir, current_team, current_aoi, "%s.config_metrics.json" % current_aoi)
            if metrics_json_filepath.is_file():
                with open(str(metrics_json_filepath.absolute())) as json_file:
                    json_data = json.load(json_file)
                # Check offset file
                offset_file_path = Path(root_dir, current_team, "%s.offset.txt" % current_aoi)
                if offset_file_path.is_file():
                    with open(str(offset_file_path.absolute())) as offset_file:
                        if offset_file_path.suffix is ".json":
                            offset_data = json.load(offset_file)
                        else:
                            offset_data = offset_file.readline()
                        n = {}
                        n["threshold_geometry"] = json_data["threshold_geometry"]
                        n["relative_accuracy"] = json_data["relative_accuracy"]
                        n["registration_offset"] = offset_data["offset"]
                        n["gelocation_error"] = np.linalg.norm(n["registration_offset"], 2)
                        n["terrain_accuracy"] = None
                        json_data = n
                        del n, offset_data

                if "terrain_accuracy" in json_data.keys():
                    n = {}
                    n["threshold_geometry"] = {}
                    n["relative_accuracy"] = {}
                    n["objectwise"] = {}
                    for cls in range(0, json_data["threshold_geometry"].__len__()):
                        current_class = json_data["threshold_geometry"][cls]['CLSValue'][0]
                        n["threshold_geometry"].update({current_class: json_data["threshold_geometry"][cls]})
                        n["relative_accuracy"].update({current_class: json_data["relative_accuracy"][cls]})
                        try:
                            n["objectwise"].update({current_class: json_data["objectwise"][cls]})
                        except KeyError:
                            print('No objectwise metrics found...')
                    n["registration_offset"] = json_data["registration_offset"]
                    n["gelocation_error"] = json_data["gelocation_error"]
                    n["terrain_accuracy"] = None
                    json_data = n
                    del n

                container = Result(current_team, current_aoi, json_data)
                if current_team not in all_results.keys():
                    all_results[current_team] = {}
                all_results[current_team].update({current_aoi: container})
            else:
                container = Result(current_team, current_aoi, "")
                all_results[current_team] = {current_aoi: container}

            # Try to find config file
            config_path = Path(root_dir, current_team, current_aoi, current_aoi + '.config')
            if config_path.is_file():
                config = geo.parse_config(str(config_path.absolute()),
                                          refpath=(ref_path or str(config_path.parent)),
                                          testpath=(test_path or str(config_path.parent)))

                # Get test model information from configuration file.
                test_dsm_filename = config['INPUT.TEST']['DSMFilename']
                test_dtm_filename = config['INPUT.TEST'].get('DTMFilename', None)
                test_cls_filename = config['INPUT.TEST']['CLSFilename']

                # Get reference model information from configuration file.
                ref_dsm_filename = config['INPUT.REF']['DSMFilename']
                ref_dtm_filename = config['INPUT.REF']['DTMFilename']
                ref_cls_filename = config['INPUT.REF']['CLSFilename']
                ref_ndx_filename = config['INPUT.REF']['NDXFilename']

                # Get plot settings from configuration file
                PLOTS_SHOW = config['PLOTS']['ShowPlots']
                PLOTS_SAVE = config['PLOTS']['SavePlots']
            elif Path(config_path.parent, config_path.stem + ".json").is_file():
                print('Old config file, parsing via json...')
                is_config = False
                config_path = Path(config_path.parent, config_path.stem + ".json")
                with open(str(config_path.absolute())) as config_file_json:
                    config = json.load(config_file_json)

                # Get test model information from configuration file.
                test_dsm_filename = config['INPUT.TEST']['DSMFilename']
                test_dtm_filename = config['INPUT.TEST'].get('DTMFilename', None)
                test_cls_filename = config['INPUT.TEST']['CLSFilename']

                # Get reference model information from configuration file.
                ref_dsm_filename = config['INPUT.REF']['DSMFilename']
                ref_dtm_filename = config['INPUT.REF']['DTMFilename']
                ref_cls_filename = config['INPUT.REF']['CLSFilename']
                ref_ndx_filename = config['INPUT.REF']['NDXFilename']

                # Get plot settings from configuration file
                PLOTS_SHOW = config['PLOTS']['ShowPlots']
                PLOTS_SAVE = config['PLOTS']['SavePlots']

    # Flatten list in case of json/config discrepencies
    if not is_config:
        config["INPUT.REF"]["CLSMatchValue"] = [item for sublist in config["INPUT.REF"]["CLSMatchValue"] for item in sublist]

    # compute averaged metrics
    averaged_results = {}
    sum_2d_completeness = 0
    sum_2d_correctness = 0
    sum_2d_jaccard_index = 0
    sum_2d_fscore = 0
    sum_3d_completeness = 0
    sum_3d_correctness = 0
    sum_3d_jaccard_index = 0
    sum_3d_fscore = 0
    sum_hmrse = 0
    sum_zrmse = 0
    sum_geolocation_error = 0
    for team in all_results:
        total_aois = all_results[team].__len__()
        averaged_results[team] = {}
        for aoi in all_results[team]:
            sum_geolocation_error = sum_geolocation_error + all_results[team][aoi].results['gelocation_error']
            for cls in all_results[team][aoi].results['threshold_geometry']:
                sum_2d_completeness = sum_2d_completeness + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['completeness']
                sum_2d_correctness = sum_2d_correctness + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['correctness']
                sum_2d_jaccard_index = sum_2d_jaccard_index + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['jaccardIndex']
                sum_2d_fscore = sum_2d_fscore + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['fscore']
                sum_3d_completeness = sum_3d_completeness + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['completeness']
                sum_3d_correctness = sum_3d_correctness + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['correctness']
                sum_3d_jaccard_index = sum_3d_jaccard_index + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['jaccardIndex']
                sum_3d_fscore = sum_3d_fscore + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['fscore']
                sum_zrmse = sum_zrmse + all_results[team][aoi].results['relative_accuracy'][cls]['zrmse']
                sum_hrmse = sum_hmrse + all_results[team][aoi].results['relative_accuracy'][cls]['hrmse']

        average_2d_completeness = np.round(sum_2d_completeness / total_aois, decimals=2)
        averaged_results[team].update({'average_2d_completeness': average_2d_completeness})
        average_2d_correctness = np.round(sum_2d_correctness / total_aois, decimals=2)
        averaged_results[team].update({'average_2d_correctness': average_2d_correctness})
        average_2d_jaccardindex = np.round(sum_2d_jaccard_index / total_aois, decimals=2)
        averaged_results[team].update({'average_2d_jaccardindex': average_2d_jaccardindex})
        average_3d_completness = np.round(sum_3d_completeness / total_aois, decimals=2)
        averaged_results[team].update({'average_3d_completness': average_3d_completness})
        average_3d_correctness = np.round(sum_3d_correctness / total_aois, decimals=2)
        averaged_results[team].update({'average_3d_correctness': average_3d_correctness})
        average_3d_jaccardindex = np.round(sum_3d_jaccard_index / total_aois, decimals=2)
        averaged_results[team].update({'average_3d_jaccardindex': average_3d_jaccardindex})
        average_zrmse = np.round(sum_zrmse / total_aois, decimals=2)
        averaged_results[team].update({'average_zrmse': average_zrmse})
        average_hrmse = np.round(sum_hrmse / total_aois, decimals=2)
        averaged_results[team].update({'average_hrmse': average_hrmse})

    # Compute aggregated metrics
    aggregated_results = {}
    for team in all_results:
        aggregated_results[team] = {}
        for aoi in all_results[team]:
            if "geolocation_error" not in aggregated_results[team]:
                aggregated_results[team]['geolocation_errors'] = \
                    [all_results[team][aoi].results['gelocation_error']]
            else:
                aggregated_results[team]['geolocation_errors'].append(all_results[team][aoi].results['gelocation_error'])
            for cls in config["INPUT.REF"]["CLSMatchValue"]:
                aggregated_results[team][cls] = {}
                for dimension in ["2D", "3D"]:
                    aggregated_results[team][cls][dimension] = {}
                    if "TP" not in aggregated_results[team][cls][dimension]:
                        aggregated_results[team][cls][dimension]["TP"] = 0
                    if "FP" not in aggregated_results[team][cls][dimension]:
                        aggregated_results[team][cls][dimension]["FP"] = 0
                    if "FN" not in aggregated_results[team][cls][dimension]:
                        aggregated_results[team][cls][dimension]["FN"] = 0
                    aggregated_results[team][cls][dimension]["TP"] = aggregated_results[team][cls][dimension]["TP"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["TP"]
                    aggregated_results[team][cls][dimension]["FP"] = aggregated_results[team][cls][dimension]["FP"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["FP"]
                    aggregated_results[team][cls][dimension]["FN"] = aggregated_results[team][cls][dimension]["FN"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["FN"]
        # After all AOIs are compiled, calculate other metrics
        for cls in config["INPUT.REF"]["CLSMatchValue"]:
            for dimension in ["2D", "3D"]:
                TP = aggregated_results[team][cls][dimension]["TP"]
                FP = aggregated_results[team][cls][dimension]["FP"]
                FN = aggregated_results[team][cls][dimension]["FN"]
                completeness = np.round(TP / (TP + FN), decimals=2)
                correctness = np.round(TP / (TP + FP), decimals=2)
                fscore = np.round((2 * completeness * correctness) / (completeness + correctness), decimals=2)
                jaccard_index = np.round(fscore / (2 - fscore), decimals=2)
                branching_factor = np.round(FP / TP, decimals=2)
                miss_factor = np.round(FN / TP, decimals=2)
                aggregated_results[team][cls][dimension]["completeness"] = completeness
                aggregated_results[team][cls][dimension]["correctness"] = correctness
                aggregated_results[team][cls][dimension]["fscore"] = fscore
                aggregated_results[team][cls][dimension]["jaccardindex"] = jaccard_index
                aggregated_results[team][cls][dimension]["branchingfactor"] = branching_factor
                aggregated_results[team][cls][dimension]["missfactor"] = miss_factor

    return averaged_results


def main():
    root_dir = Path(r"C:\Users\wangss1\Documents\Data\ARA_Metrics_Dry_Run")
    teams = [r'ARA']
    aois = [r'AOI_D4']
    baa_threshold = BAAThresholds()
    summarized_results = summarize_metrics(root_dir, teams, aois)


if __name__ == "__main__":
    main()













