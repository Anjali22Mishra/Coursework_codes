# ML Coursework Notebooks

Notebooks from machine-learning coursework completed at IIT Madras. This is a private
repository, shared with reviewers on request.

> These are **course assignments**, not research work. Course handouts and datasets are not
> included; each folder links to the public source of its data.

| Folder | Topic | Summary |
|---|---|---|
| [`01_mnist_neural_network_numpy`](01_mnist_neural_network_numpy) | Neural networks from scratch (NumPy) on MNIST digits | 1- and 2-hidden-layer networks trained with gradient descent; 87.3% and 90.9% accuracy on 1,000 held-out images |
| [`02_logistic_regression_gd_vs_sgd`](02_logistic_regression_gd_vs_sgd) | Logistic regression with full-batch GD vs SGD on stock-market data | Convergence behaviour, test misclassifications, and a label-shift experiment |

## Running the notebooks

Both notebooks were written for Google Colab (Python 3, NumPy, pandas, Matplotlib). To run them
locally, install `numpy pandas matplotlib jupyter`, download the data as described in each
folder's README, and update the file paths in the first cells.

## Rights

Copyright (c) 2026 <Anjali Mishra>. All rights reserved. This repository is shared for viewing and
evaluation only. No permission is granted to copy, modify or redistribute its contents without
the author's written consent.
