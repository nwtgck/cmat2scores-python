import numpy as np
import cmat2scores

cmat = np.array([
  [3, 2, 0],
  [2, 1, 0],
  [0, 0, 3]
])

# Calc accuracy
accuracy       = cmat2scores.accuracy_score(cmat)

# Calc recall
# (NOTE: You can use all argument which are defined in sklearn.metrics.recall_score())
recall   = cmat2scores.recall_score(cmat, average='macro')

print('Accuracy:', accuracy)
print('Recall:', recall)

