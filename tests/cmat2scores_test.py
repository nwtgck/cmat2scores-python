import unittest
from sklearn import metrics
import numpy as np
import cmat2scores
from sklearn import datasets
from sklearn import neural_network
from sklearn import model_selection

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

  # Get a dataset
  X, y = datasets.load_digits(return_X_y=True)
  # Split into train & test
  X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size = 0.5, random_state = 103
  )
  # Create a classifier
  clf = neural_network.MLPClassifier()
  # Fit
  clf.fit(X_train, y_train)
  # Predict test set
  y_pred = clf.predict(X_test)
  # Get confusion matrix
  cmat = metrics.confusion_matrix(y_test, y_pred)

  # Accuracy test
  actual = cmat2scores.accuracy_score(cmat)
  expect = metrics.accuracy_score(y_test, y_pred)
  assert(actual == expect)

  # Precision test
  actual = cmat2scores.precision_score(cmat, average='macro')
  expect = metrics.precision_score(y_test, y_pred, average='macro')
  assert (actual == expect)

  # Recall test
  actual = cmat2scores.recall_score(cmat, average='macro')
  expect = metrics.recall_score(y_test, y_pred, average='macro')
  assert (actual == expect)

  # F1 test
  actual = cmat2scores.f1_score(cmat, average='macro')
  expect = metrics.f1_score(y_test, y_pred, average='macro')
  assert (actual == expect)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(Cmat2scoresTest))
  return suite