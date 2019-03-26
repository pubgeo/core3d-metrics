import json
import jsonschema
import numpy as np
from pathlib import Path
from MetricContainer import Result

# BAA Thresholds
class baa_thresholds:

    def __init__(self):
        self.geolocation_error = np.array([2, 1.5, 1.5, 1])*3.5
        self.completeness_2d = np.array([0.8, 0.85, 0.9, 0.95])
        self.correctness_2d = np.array([0.8, 0.85, 0.9, 0.95])
        self.completeness_3d = np.array([0.6, 0.7, 0.8, 0.9])
        self.correctness_3d = np.array([0.6, 0.7, 0.8, 0.9])
        self.material_accuracy = np.array([0.85, 0.90, 0.95, 0.98])
        self.model_build_time = np.array([8, 2, 2, 1])
        self.fscore_2d = 2*self.completeness_2d * self.correctness_2d / self.completeness_2d + self.correctness_2d
        self.fscore_3d = 2*self.completeness_3d * self.correctness_3d / self.completeness_3d + self.correctness_3d
        self.jaccard_index_2d = self.fscore_2d / (2-self.fscore_2d)
        self.jaccard_index_3d = self.fscore_3d / (2-self.fscore_3d)


def summarize_data(baa_threshold):
    # load results8
    root_dir = Path(r"\\dom1\Core\Dept\AOS\AOSShare\IARPA-CORE3D\Workspace\Leichtman\2018-10-01-Performer-Eval")
    teams = ['ARA', 'GERC', 'KW', 'VSI']
    tests = ['2018-06-04-Self-Test', '2018-10-02-As-Delivered', '2018-10-10-APL-Run'];
    aois = ['D1', 'D2', 'D3', 'D4']
    all_results = []
    for current_team in teams:
        for current_test in tests:
            for current_aoi in aois:
                json_file_path = Path(root_dir, current_team, current_test, "%s.config_metrics.json" % current_aoi)
                if json_file_path.is_file():
                    with open(str(json_file_path.absolute())) as json_file:
                        json_data = json.load(json_file)
                    # Check offset file
                    offset_file_path = Path(root_dir, current_team, current_test, "%s.offset.json" % current_aoi)
                    if offset_file_path.is_file():
                        with open(str(offset_file_path.absolute())) as offset_json_file:
                            offset_data = json.load(offset_json_file)
                            n = {}
                            n["threshold_geometry"] = json_data["threshold_geometry"]
                            n["relative_accuracy"] = json_data["relative_accuracy"]
                            n["registration_offset"] = offset_data["offset"]
                            n["gelocation_error"] = np.linalg.norm(n["registration_offset"], 2)
                            n["terrain_accuracy"] = None
                            n["threshold_materials"] = json_data["threshold_materials"]
                            json_data = n
                            del n, offset_data

                    if "terrain_accuracy" in json_data.keys():
                        n = {}
                        n["threshold_geometry"] = json_data["threshold_geometry"]
                        n["relative_accuracy"] = json_data["relative_accuracy"]
                        n["registration_offset"] = json_data["registration_offset"]
                        n["gelocation_error"] = json_data["gelocation_error"]
                        n["terrain_accuracy"] = None
                        n["threshold_materials"] = json_data["threshold_materials"]
                        json_data = n
                        del n

                    container = Result(current_team, current_test, current_aoi, json_data)
                    all_results.append(container)
                else:
                    container = Result(current_team, current_test, current_aoi, "")
                    all_results.append(container)

                # Try to find config file
    print("hi")

def main():
    baa_threshold = baa_thresholds()
    summarize_data(baa_threshold)
    


if __name__ == "__main__":
    main()













