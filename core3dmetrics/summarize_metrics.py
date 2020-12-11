import json
import jsonschema
import numpy as np
import csv
import glob
import os
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
        self.spherical_error = np.array([.86, .86, .86, .86])
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
    all_config = {}
    # Parse results
    for current_team in teams:
        for current_aoi in aois:
            metrics_json_filepath = None
            current_dir =  Path(root_dir, current_team, current_aoi)
            for file in glob.glob(os.path.join(current_dir, "*.config_metrics.json")):
                results_path = file
                metrics_json_filepath = Path(results_path)
            # metrics_json_filepath = Path(root_dir, current_team, current_aoi, "%s.config_metrics.json" % current_aoi)
            if metrics_json_filepath.is_file():
                with open(str(metrics_json_filepath.absolute())) as json_file:
                    json_data = json.load(json_file)
                # Check offset file
                current_dir = Path(root_dir, current_team, current_aoi)
                offset_file_path = None
                for file in glob.glob(os.path.join(current_dir, "*offsets.txt")):
                    offset_file_path = file
                    offset_file_path = Path(offset_file_path)
                # offset_file_path = Path(root_dir, current_team, "%s.offset.txt" % current_aoi)
                if offset_file_path.is_file():
                    with open(str(offset_file_path.absolute())) as offset_file:
                        if offset_file_path.suffix is ".json":
                            offset_data = json.load(offset_file)
                        else:
                            offset_data = {}
                            for last_line in offset_file:
                                try:
                                    offset_data["offset"] = [float(idx) for idx in last_line.split()]
                                except ValueError:
                                    continue
                        n = {}
                        n["threshold_geometry"] = json_data["threshold_geometry"]
                        n["relative_accuracy"] = json_data["relative_accuracy"]
                        n["registration_offset"] = offset_data["offset"]
                        n["geolocation_error"] = np.linalg.norm(n["registration_offset"], 2)
                        n["terrain_accuracy"] = None
                        json_data = n
                        del n, offset_data

                if "terrain_accuracy" in json_data.keys():
                    n = {}
                    n["threshold_geometry"] = {}
                    n["relative_accuracy"] = {}
                    n["objectwise"] = {}
                    classes_skipped = 0
                    for cls in range(0, json_data["threshold_geometry"].__len__()):
                        current_class = json_data["threshold_geometry"][cls]['CLSValue'][0]
                        if np.isnan(json_data["threshold_geometry"][cls]['2D']['fscore']):
                            classes_skipped = classes_skipped+1
                            continue
                        n["threshold_geometry"].update({current_class: json_data["threshold_geometry"][cls]})
                        n["relative_accuracy"].update({current_class: json_data["relative_accuracy"][cls-classes_skipped]})
                        try:
                            n["objectwise"].update({current_class: json_data["objectwise"][cls]})
                        except KeyError:
                            print('No objectwise metrics found...')
                        except IndexError:
                            print('Classification doesnt exist...')
                    n["registration_offset"] = json_data["registration_offset"]
                    n["geolocation_error"] = json_data["geolocation_error"]
                    n["terrain_accuracy"] = None
                    try:
                        n["instance_f1"] = json_data["objectwise"][cls]["instance_f1"]
                        n["instance_f1_merge_fp"] = json_data["objectwise"][cls]["instance_f1_merge_fp"]
                        n["instance_f1_merge_fn"] = json_data["objectwise"][cls]["instance_f1_merge_fn"]
                    except KeyError:
                        n["instance_f1"] = np.nan
                        n["instance_f1_merge_fp"] = np.nan
                        n["instance_f1_merge_fn"] = np.nan
                    except IndexError:
                        n["instance_f1"] = np.nan
                        n["instance_f1_merge_fp"] = np.nan
                        n["instance_f1_merge_fn"] = np.nan
                        print('Classification doesnt exist...')

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
            current_dir = Path(root_dir, current_team, current_aoi)
            config_path = None
            for file in glob.glob(os.path.join(current_dir,"*.config")):
                config_path = file
                config_path = Path(config_path)
            # config_path = Path(root_dir, current_team, current_aoi, current_aoi + '.config')
            if config_path.is_file():
                config = geo.parse_config(str(config_path.absolute()),
                                          refpath=(ref_path or str(config_path.parent)),
                                          testpath=(test_path or str(config_path.parent)))
            elif Path(config_path.parent, config_path.stem + ".json").is_file():
                print('Old config file, parsing via json...')
                is_config = False
                config_path = Path(config_path.parent, config_path.stem + ".json")
                with open(str(config_path.absolute())) as config_file_json:
                    config = json.load(config_file_json)

            # Flatten list in case of json/config discrepencies
            if not is_config:
                config["INPUT.REF"]["CLSMatchValue"] = [item for sublist in config["INPUT.REF"]["CLSMatchValue"]
                                                        for item in sublist]
            # Store config for each aoi
            if current_team not in all_config.keys():
                all_config[current_team] = {}
            if current_aoi not in all_config.keys():
                all_config[current_team][current_aoi] = {}
            all_config[current_team][current_aoi] = config
            all_config[current_team][current_aoi].update({'path': config_path})

    # compute averaged metrics and write out objectwise metrics
    with open('objectwise_metrics.csv', mode='w') as objectwise_file:
        csv_writer = csv.writer(objectwise_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Building Index', '2D IOU', '3D IOU', 'HRMSE', 'ZRMSE'])
        averaged_results = {}
        for team in all_results:
            sum_2d_completeness = {}
            sum_2d_correctness = {}
            sum_2d_jaccard_index = {}
            sum_2d_fscore = {}
            sum_3d_completeness = {}
            sum_3d_correctness = {}
            sum_3d_jaccard_index = {}
            sum_3d_fscore = {}
            sum_geolocation_error = 0
            sum_hrmse = {}
            sum_zrmse = {}
            averaged_results[team] = {}
            evaluated_classes = []
            for aoi in all_results[team]:
                sum_geolocation_error = sum_geolocation_error + all_results[team][aoi].results["geolocation_error"]
                for cls in all_results[team][aoi].results["threshold_geometry"]:
                    evaluated_classes.append(cls)
                    if cls not in sum_2d_completeness.keys():
                        sum_2d_completeness[cls] = 0
                        sum_2d_correctness[cls] = 0
                        sum_2d_jaccard_index[cls] = 0
                        sum_2d_fscore[cls] = 0
                        sum_3d_completeness[cls] = 0
                        sum_3d_correctness[cls] = 0
                        sum_3d_jaccard_index[cls] = 0
                        sum_3d_fscore[cls] = 0
                        sum_zrmse[cls] = 0
                        sum_hrmse[cls] = 0
                    sum_2d_completeness[cls] = sum_2d_completeness[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['completeness']
                    sum_2d_correctness[cls] = sum_2d_correctness[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['correctness']
                    sum_2d_jaccard_index[cls] = sum_2d_jaccard_index[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['jaccardIndex']
                    sum_2d_fscore[cls] = sum_2d_fscore[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['2D']['fscore']
                    sum_3d_completeness[cls] = sum_3d_completeness[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['completeness']
                    sum_3d_correctness[cls] = sum_3d_correctness[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['correctness']
                    sum_3d_jaccard_index[cls] = sum_3d_jaccard_index[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['jaccardIndex']
                    sum_3d_fscore[cls] = sum_3d_fscore[cls] + all_results[team][aoi].results['threshold_geometry'][cls]['3D']['fscore']
                    sum_hrmse[cls] = sum_hrmse[cls] + all_results[team][aoi].results['relative_accuracy'][cls]["hrmse"]
                    sum_zrmse[cls] = sum_zrmse[cls] + all_results[team][aoi].results['relative_accuracy'][cls]["zrmse"]
            # Average results for evaluated classes in config file
            averaged_results[team]["geolocation_error"] = np.round(
                sum_geolocation_error / all_results[team].__len__(), decimals=2)
            # TODO: Need to make config specific to each config file, but for now it doesn't matter
            for cls in np.unique(evaluated_classes):
                try:
                    averaged_results[team][cls] = {}
                    averaged_results[team][cls]["2d_completeness"] = np.round(
                        sum_2d_completeness[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["2d_correctness"] = np.round(
                        sum_2d_correctness[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["2d_jaccard_index"] = np.round(
                        sum_2d_jaccard_index[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["2d_fscore"] = np.round(
                        sum_2d_fscore[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["3d_completeness"] = np.round(
                        sum_3d_completeness[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["3d_correctness"] = np.round(
                        sum_3d_correctness[cls] /evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["3d_jaccard_index"] = np.round(
                        sum_3d_jaccard_index[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["fscore"] = np.round(
                        sum_3d_fscore[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["hrmse"] = np.round(
                        sum_hrmse[cls] / evaluated_classes.count(cls), decimals=2)
                    averaged_results[team][cls]["zrmse"] = np.round(
                        sum_zrmse[cls] / evaluated_classes.count(cls), decimals=2)
                except KeyError:
                    print('Class not found, skipping...')
                    continue

    return averaged_results, all_results, all_config


def main():
    print("For Debug")


if __name__ == "__main__":
    main()













