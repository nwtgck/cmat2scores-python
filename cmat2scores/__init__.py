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
