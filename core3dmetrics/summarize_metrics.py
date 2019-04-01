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
        self.jaccard_index_2d = self.fscore_2d / (2-self.fscore_2d)
        self.jaccard_index_3d = self.fscore_3d / (2-self.fscore_3d)


def summarize_data(baa_threshold, root_dir, teams, aois, ref_path=None, test_path=None):
    # load results8
    all_results = {}
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
                        n["objectwise"].update({current_class: json_data["objectwise"][cls]})
                    n["registration_offset"] = json_data["registration_offset"]
                    n["gelocation_error"] = json_data["gelocation_error"]
                    n["terrain_accuracy"] = None
                    json_data = n
                    del n

                container = Result(current_team, current_aoi, json_data)
                all_results[current_team] = {current_aoi: container}
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

                # Get material label names and list of material labels to ignore in evaluation.
                material_names = config['MATERIALS.REF']['MaterialNames']
                material_indices_to_ignore = config['MATERIALS.REF']['MaterialIndicesToIgnore']

                # Get plot settings from configuration file
                PLOTS_SHOW = config['PLOTS']['ShowPlots']
                PLOTS_SAVE = config['PLOTS']['SavePlots']

    # Computer combined metrics
    summarized_results = {}
    for team in all_results:
        summarized_results[team] = {}
        for aoi in all_results[team]:
            for cls in config["INPUT.REF"]["CLSMatchValue"]:
                summarized_results[team][cls] = {}
                for dimension in ["2D", "3D"]:
                    summarized_results[team][cls][dimension] = {}
                    if "TP" not in summarized_results[team][cls][dimension]:
                        summarized_results[team][cls][dimension]["TP"] = 0
                    if "FP" not in summarized_results[team][cls][dimension]:
                        summarized_results[team][cls][dimension]["FP"] = 0
                    if "FN" not in summarized_results[team][cls][dimension]:
                        summarized_results[team][cls][dimension]["FN"] = 0
                    summarized_results[team][cls][dimension]["TP"] = summarized_results[team][cls][dimension]["TP"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["TP"]
                    summarized_results[team][cls][dimension]["FP"] = summarized_results[team][cls][dimension]["FP"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["FP"]
                    summarized_results[team][cls][dimension]["FN"] = summarized_results[team][cls][dimension]["FN"] + \
                                                                all_results[team][aoi].results["threshold_geometry"][cls][
                                                                    dimension]["FN"]
        # After all AOIs are compiled, calculate other metrics
        for cls in config["INPUT.REF"]["CLSMatchValue"]:
            for dimension in ["2D", "3D"]:
                TP = summarized_results[team][cls][dimension]["TP"]
                FP = summarized_results[team][cls][dimension]["FP"]
                FN = summarized_results[team][cls][dimension]["FN"]
                completeness = TP / (TP + FN)
                correctness = TP / (TP + FP)
                fscore = (2 * completeness * correctness) / (completeness + correctness)
                jaccard_index = fscore / (2 - fscore)
                branching_factor = FP / TP
                miss_factor = FN / TP
                summarized_results[team][cls][dimension]["completeness"] = completeness
                summarized_results[team][cls][dimension]["correctness"] = correctness
                summarized_results[team][cls][dimension]["fscore"] = fscore
                summarized_results[team][cls][dimension]["jaccardindex"] = jaccard_index
                summarized_results[team][cls][dimension]["branchingfactor"] = branching_factor
                summarized_results[team][cls][dimension]["missfactor"] = miss_factor

    return summarized_results


def main():
    root_dir = Path(r"C:\Users\wangss1\Documents\Data\ARA_Metrics_Dry_Run")
    teams = [r'ARA']
    aois = [r'AOI_D4']
    baa_threshold = BAAThresholds()
    summarized_results = summarize_data(baa_threshold, root_dir, teams, aois)


if __name__ == "__main__":
    main()













