# cmat2scores

[![Build Status](https://travis-ci.org/nwtgck/cmat2scores-python.svg?branch=develop)](https://travis-ci.org/nwtgck/cmat2scores-python) [![Coverage Status](https://coveralls.io/repos/github/nwtgck/cmat2scores-python/badge.svg?branch=develop)](https://coveralls.io/github/nwtgck/cmat2scores-python?branch=develop) 

Calculate accuracy, precision, recall and f-measure from confusion matrix

## Installation (pip)

```bash
pip3 install git+https://github.com/nwtgck/cmat2scores-python
```
## Installation (Pipenv)

```bash
pipenv install git+https://github.com/nwtgck/cmat2scores-python.git@master#egg=cmat2scores
```

## Usage

```python
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
```
