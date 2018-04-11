import unittest
import numpy as np

import core3dmetrics.geometrics as geo


class TestGeometryMetrics(unittest.TestCase):

  def setUp(self):
    pass

  # common test (unittest does not run, as there is no "test" prefix) 
  def common_test(self,reffac,testfac,metrics_expected):

    # base inputs
    DSM = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],dtype=np.float32)
    sh = DSM.shape

    DTM = np.zeros(sh,dtype=np.float32)
    MSK = (DSM!=0)

    tform = [0,.5,0,0,0,.5]
    ignore = np.zeros(sh,dtype=np.bool)

    # calculate metrics
    metrics = geo.run_threshold_geometry_metrics(
      reffac*DSM, DTM, MSK, testfac*DSM, DTM, MSK, tform, ignore,
      plot=None,verbose=True)

    # compare subset of metrics
    for section in metrics_expected:
      metrics_subset = {k:metrics[section].get(k) for k in metrics_expected[section]}
      self.assertDictEqual(metrics_subset, metrics_expected[section],
          '{} metrics are not as expected'.format(section))


  # ref/test both above ground
  def test_both_positive(self):
    metrics_expected = {
      "2D": {"TP": 4.0, "FN": 0.0, "FP": 0.0},
      "3D": {"TP": 1.0, "FN": 0.0, "FP": 0.0},
    }
    self.common_test(1.0,1.0,metrics_expected)

  # ref-test both below ground
  def test_both_negative(self):
    metrics_expected = {
      "2D": {"TP": 4.0, "FN": 0.0, "FP": 0.0},
      "3D": {"TP": 1.0, "FN": 0.0, "FP": 0.0},
    }
    self.common_test(-1.0,-1.0,metrics_expected)

  # testDSM below ground
  def test_negative_testDSM(self):
    metrics_expected = {
      "2D": {"TP": 0.0, "FN": 4.0, "FP": 4.0},
      "3D": {"TP": 0.0, "FN": 1.0, "FP": 1.0},
    }
    self.common_test(1.0,-1.0,metrics_expected)

  # refDSM below ground
  def test_negative_refDSM(self):
    metrics_expected = {
      "2D": {"TP": 0.0, "FN": 4.0, "FP": 4.0},
      "3D": {"TP": 0.0, "FN": 1.0, "FP": 1.0},
    }
    self.common_test(-1.0,1.0,metrics_expected)




if __name__ == '__main__':
  unittest.main()