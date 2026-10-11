# EMPL

PyTorch implementation of **Embedding-driven Meta-Parameter Learning (EMPL)** for motor-imagery EEG decoding.

## Structure

```
empl/
  embeddings.py   sample-wise hybrid temporal embedding E_t = E_proto + alpha * E_phys, spatial embedding E_s
  meta.py         balanced subspace sampling and group orthogonality loss
  temporal.py     meta-temporal encoder (MHSA + embedding-driven meta-causal convolution, sliding windows)
  spatial.py      meta-spatial encoder (adaptive adjacency from E_s, node-specific Meta-GCRU)
  model.py        EMPL
data.py           BCI Competition IV-2a loading via MOABB, 1.5-6.0 s window, z-scoring with training-set statistics
train.py          within-subject and leave-one-subject-out (LOSO) training and evaluation
```

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x, MNE, MOABB, scikit-learn. The BCI-IV-2a data are downloaded by MOABB on first use.

## Usage

Within-subject (session 1 for training, session 2 for testing, 1000 epochs, five runs per subject):

```
python train.py --protocol within --subjects 1 2 3 4 5 6 7 8 9
```

Leave-one-subject-out (eight source subjects for training and validation, target subject for testing, 125 epochs):

```
python train.py --protocol loso --subjects 1 2 3 4 5 6 7 8 9
```

Results are written to `results/<protocol>.json` (accuracy and Cohen's kappa per run and their means).

## Default settings

Adam, learning rate 9e-4, batch size 64, loss = cross-entropy + 0.2 * group orthogonality loss;
d_t = d_s = 64, N_g = 8, tau = 1, J = 2, alpha = 0.3, M = 5 sliding windows of length 16 with stride 1.
