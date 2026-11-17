# ANAM model spec

Notation: own symbols, defined once here. Cite paper equation numbers in brackets.

## 1. Data and likelihood
- Response, exposure, offset:
- Frequency: Poisson, log link — mean and NLL:
- Synthetic: Gamma parametrisation (shape, scale), mean, variance:

## 2. Linear predictor
- Intercept, offset, main effects s_i, interactions h_jk, alpha_1 / alpha_2:
- Centring rule (what is subtracted, computed on what):

## 3. Subnetworks
- Main effect net: layers, activation, init:
- Pair net:
- Categorical terms:

## 4. Penalties
- Smoothness: grid, finite difference, absolute vs squared (paper's choice):
- Marginal clarity: formula, computed per batch or per epoch:

## 5. Monotonicity
- Calibrator: keypoints, how kept monotone:
- 1-D and 2-D lattice: interpolation formula:
- Constraint set and projection (Dykstra, iterations):

## 6. Importance and selection
- Importance score formula:
- Stage 1 / 2 / 3 procedure and stopping rules:

## 7. Training
- Optimiser, batch size, epochs, early stopping:

## 8. Gaps — what the paper does not say
| Item | Paper says | Authors' notebook says | My choice |
| --- | --- | --- | --- |
| beta (synthetic) | | | |
| alpha_1, alpha_2 init/constraint | | | |
| learning rates | | | |
| penalty weights | | | |
| NLL definition (mean? constants?) | | | |
