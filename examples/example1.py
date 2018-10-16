import numpy as np
import cmat2scores

cmat = np.array([
  [3, 2, 0],
  [2, 1, 0],
  [0, 0, 3]
])

# Calculate accuracy 
accuracy = cmat2scores.accuracy_score(cmat)
# Calculate precision
precision = cmat2scores.precision_score(cmat, average='macro')
# Calculate recall
recall    = cmat2scores.recall_score(cmat, average='macro')
# Calculate f1
f1        = cmat2scores.f1_score(cmat, average='macro')

print('Accuracy:', accuracy)
print('Precision', precision)
print('Recall:', recall)
print('F1', f1)
