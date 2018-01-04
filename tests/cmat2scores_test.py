import unittest
from sklearn import metrics
import numpy as np
import cmat2scores

class Cmat2scoresTest(unittest.TestCase):
  cmat = np.array([
    [3, 2, 0],
    [2, 1, 0],
    [0, 0, 3]
  ])
  # Create y_true and y_pred
  psuedo_y_true, psuedo_y_pred = cmat2scores.cmat_to_psuedo_y_true_and_y_pred(cmat)
  # Calc cmat by sklearn
  sklearn_cmat = metrics.confusion_matrix(psuedo_y_true, psuedo_y_pred)
  # Should be equal
  assert(np.array_equal(cmat, sklearn_cmat))


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(Cmat2scoresTest))
  return suite