import unittest
import numpy as np

import core3dmetrics.geometrics as geo


class TestGeometryMetrics(unittest.TestCase):

  def setUp(self):
    pass

  # common  test (unittest does not run, as there is no "test" prefix) 
  def common_test(self,reffac,testfac,testshft,metrics_expected):

    # base inputs
    DSM = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.float32)
    sh = DSM.shape
    DTM = np.zeros(sh,dtype=np.float32)

    tform = [0,.5,0,0,0,.5]
    ignore = np.zeros(sh,dtype=np.bool)

    # ref DSM & footprint
    refDSM = reffac*DSM
    refMSK = (refDSM!=0)

    # test DSM & footprint
    testDSM = testfac*DSM
    testDSM = np.roll(testDSM,testshft[0],axis=0)
    testDSM = np.roll(testDSM,testshft[1],axis=1)
    testMSK = (testDSM!=0)

    # calculate metrics
    metrics = geo.run_threshold_geometry_metrics(
      refDSM, DTM, refMSK, testDSM, DTM, testMSK, tform, ignore,
      plot=None,verbose=False)

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
    self.common_test(1.0,1.0,(0,0),metrics_expected)

  # ref-test both below ground
  def test_both_negative(self):
    metrics_expected = {
      "2D": {"TP": 4.0, "FN": 0.0, "FP": 0.0},
      "3D": {"TP": 1.0, "FN": 0.0, "FP": 0.0},
    }
    self.common_test(-1.0,-1.0,(0,0),metrics_expected)

  # testDSM below ground
  def test_negative_testDSM(self):
    metrics_expected = {
      "2D": {"TP": 4.0, "FN": 0.0, "FP": 0.0},
      "3D": {"TP": 0.0, "FN": 1.0, "FP": 1.0},
    }
    self.common_test(1.0,-1.0,(0,0),metrics_expected)

  # refDSM below ground
  def test_negative_refDSM(self):
    metrics_expected = {
      "2D": {"TP": 4.0, "FN": 0.0, "FP": 0.0},
      "3D": {"TP": 0.0, "FN": 1.0, "FP": 1.0},
    }
    self.common_test(-1.0,1.0,(0,0),metrics_expected)

  # shift testDSM
  def test_shift_testDSM(self):
    metrics_expected = {
      "2D": {"TP": 1.0, "FN": 3.0, "FP": 3.0},
      "3D": {"TP": 0.25, "FN": 0.75, "FP": 0.75},
    }
    self.common_test(1.0,1.0,(1,1),metrics_expected)




if __name__ == '__main__':
  unittest.main()

