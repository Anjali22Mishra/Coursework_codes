# Logistic regression: full-batch gradient descent vs SGD

Logistic regression implemented from scratch in NumPy and trained with (a) gradient descent using the full gradient and (b) stochastic gradient descent, on the *Smarket* daily stock-market data.

Context: coursework at IIT Madras. Function names and signatures follow the template given with the assignment.

## What the notebook does

**1. Data preparation**
- Loads the Smarket data (S&P 500 daily returns, 2001-2005).
- Builds the design matrix from `Lag1`-`Lag5` (returns of the previous five days), `Volume` and `Today` (the same day's return), with a column of ones added for the bias term.
- Target: `Direction`, encoded as 1 for "Up" and 0 for "Down".
- Training set: all observations before 2005. Test set: observations from 2005.
- Computes the correlation matrix of the variables.

**2. Model and optimisation routines**
- `sigmoid(z)`: the logistic function.
- `fullGradient(w, X, y)`: the logistic-loss gradient summed over all training samples.
- `randGradient(w, X, y, n)`: the gradient for the single sample with index `n`.
- `rho(t)`: the step-size schedule, 1/sqrt(t).
- `fdescent(X, y, rho, job, nEpoch)`: the training routine. Weights start at zero. In each epoch it loops over the sample indices (in order for `'full'`, in a random permutation for `'stochastic'`). Each update uses the full gradient (`'full'`) or the single-sample gradient (`'stochastic'`) and step size `rho(epoch * N + i + 1)`. After every update it records the mean cross-entropy objective on the training set (probabilities clipped at 1e-8 inside the logarithm). It returns the final weights and the objective history.
- `predictions(X, y, w)`: predicts "Up" when the predicted probability is at least 0.5 and counts the misclassified points.

**3. Experiments**
- Trains both methods for 3 epochs and plots the objective value against the number of updates for each.
- Predicts the 2005 test set with each set of weights and counts the misclassified points.
- Shifted-label experiment: pairs the features of day *t* with the direction of day *t+1* (`X[:-1]`, `y[1:]`), re-runs both optimisers on the shifted training data and counts misclassified points on the shifted 2005 test set.

## Results printed by the notebook

Misclassified points on the 2005 test set:

| Setting | Full-gradient descent | SGD |
|---|---|---|
| Original labels | 0 | 3 |
| Labels shifted by one day | 140 | 130 |

## Data and how to run

The data is the *Smarket* dataset from the ISLR book (James, Witten, Hastie and Tibshirani, *An Introduction to Statistical Learning*), available from the book's website and in the `ISLR` R package. The CSV file is not included in this repository.

To run the notebook, place the CSV next to it, update the file path in the data-loading cell, and install `numpy`, `pandas`, `matplotlib` and `jupyter` (or open it in Google Colab).