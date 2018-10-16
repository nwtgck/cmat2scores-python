import numpy as np
import cmat2scores

# Confusion matrix
cmat = np.array([
  [6, 1, 3],
  [0, 9, 1],
  [9, 0, 10]
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
