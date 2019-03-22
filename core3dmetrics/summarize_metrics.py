import json
import jsonschema
import numpy as np
from pathlib import Path

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


def main():
    baa_threshold = baa_thresholds()
    # load results
    root_dir = Path("\\dom1\Core\Dept\AOS\AOSShare\IARPA-CORE3D\Workspace\Leichtman\2018-10-01-Performer-Eval")
    teams = ['ARA', 'GERC', 'KW', 'VSI']
    aois = ['d1', 'd2', 'd3', 'd4', 'u1']
    


if __name__ == "__main__":
    main()













