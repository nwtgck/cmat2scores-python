import numpy as np
import cmat2scores

cmat = np.array([
  [3, 2, 0],
  [2, 1, 0],
  [0, 0, 3]
])

accuracy = cmat2scores.accuracy_score(cmat)
recall   = cmat2scores.recall_score(cmat)

print(accuracy)
print(recall)

