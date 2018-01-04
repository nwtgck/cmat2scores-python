from sklearn import metrics

def cmat_to_psuedo_y_true_and_y_pred(cmat):
  """
  Convert a confusion matrix to psuedo y_true and y_pred
  :param cmat: Confusion matrix
  :return: psuedo y_true and y_pred
  """
  y_true = []
  y_pred = []
  for true_class, row in enumerate(cmat):
    for pred_class, elm in enumerate(row):
      y_true.extend([true_class] * elm)
      y_pred.extend([pred_class] * elm)
  return y_true, y_pred


def accuracy_score(cmat, **kwargs):
  """
  Accuracy classification score.
  :param cmat: Confusion matrix
  :param kwargs:
  :return: accuracy
  """
  # Create psuedo y_true and y_pred
  psuedo_y_true, psuedo_y_pred = cmat_to_psuedo_y_true_and_y_pred(cmat)
  return metrics.accuracy_score(psuedo_y_true, psuedo_y_pred, **kwargs)

def precision_score(cmat, **kwargs):
  """
  Precision classification score.
  :param cmat: Confusion matrix
  :param kwargs:
  :return: precision
  """
  # Create psuedo y_true and y_pred
  psuedo_y_true, psuedo_y_pred = cmat_to_psuedo_y_true_and_y_pred(cmat)
  return metrics.precision_score(psuedo_y_true, psuedo_y_pred, **kwargs)

def recall_score(cmat, **kwargs):
  """
  Recall classification score.
  :param cmat: Confusion matrix
  :param kwargs:
  :return: recall
  """
  # Create psuedo y_true and y_pred
  psuedo_y_true, psuedo_y_pred = cmat_to_psuedo_y_true_and_y_pred(cmat)
  return metrics.recall_score(psuedo_y_true, psuedo_y_pred, **kwargs)