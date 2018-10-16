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


```

## My Note

[MYNOTE.md](MYNOTE.md)
