# Neural networks from scratch on MNIST (NumPy)

Handwritten-digit classification with fully-connected networks implemented using only NumPy
(no TensorFlow / PyTorch): forward pass, back-propagation and gradient descent are written by hand.

Context: coursework at IIT Madras.

## What's in the notebook

| Model | Architecture | Training | Train acc.* | Held-out acc. |
|---|---|---|---|---|
| Small | 784 -> 10 (ReLU) -> 10 (softmax) | 500 iterations, lr 0.1 | 85.2% | 87.3% |
| Larger | 784 -> 20 (ReLU) -> 10 (ReLU) -> 10 (softmax) | 1000 iterations, lr 0.1 | 88.8% | 90.9% |

\*Training accuracy is the value logged at the last iteration during training.
Held-out accuracy is measured on 1,000 images kept aside from the 42,000 labelled images
(41,000 for training), with pixel values scaled to [0, 1].

The notebook also visualises individual predictions next to the true labels.

**Caveat on the comparison.** The larger network is both bigger *and* trained for twice as many
iterations, and each model was trained once with one random split and no fixed seed, so the
improvement should not be attributed to depth alone.

## Implementation

- Initialisation: uniform in [-0.5, 0.5].
- Forward: `Z = W A_prev + b`, ReLU in hidden layers, softmax at the output.
- Loss gradient at the output: `dZ_out = A_out - one_hot(Y)` (softmax + cross-entropy).
- Back-propagation through ReLU using the indicator `Z > 0`; parameter update by plain
  (full-batch) gradient descent.

## Data and how to run

The data is the Kaggle *Digit Recognizer* competition set (`train.csv`), which is **not** included
here: <https://www.kaggle.com/competitions/digit-recognizer>.

In Colab, the notebook downloads it with the Kaggle API (it expects a `kaggle.json` token in your
Google Drive; never commit that file). Alternatively, download `train.csv` manually and change the
path in the `pd.read_csv(...)` cell.

## Known limitations

- Bias gradients are computed as a single scalar (`np.sum(dZ)`) shared by all units; the correct
  gradient is one value per unit (`np.sum(dZ, axis=1, keepdims=True)`).
- Softmax is computed without subtracting the maximum, which can overflow for large logits.
- No fixed random seed, no separate test set, full-batch updates only.
- The second network is a copy of the first with a `large_` prefix; a single configurable
  network class would be cleaner.

## Acknowledgements

The overall structure (function names and workflow) follows the widely used Kaggle tutorial
"Simple MNIST NN from scratch (numpy, no TF/Keras)" by Samson Zhang. The two-hidden-layer
variant and the comparison between the two models are my additions.
