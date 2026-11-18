# ANAM model spec

## 1. Data and likelihood
- Response: nclaims for belgian data, continuous Y for synthetic data
- Exposure: lambda is exposure to risk, to account for different policies which have different exposures to risk
- Offset: log(lambda) is the offset term, to adjust for varying times between policies

- Frequency: Poisson, log link
     mean = 
     NLL = -(1/n)*sum(log(pdf(y)))
- Synthetic: Gamma parametrisation (shape, scale)
     mean = scale/shape
     variance: scale/shape^2

## 2. Linear predictor
- Linear predictor: β + offset + α₁·Σ sᵢ(xᵢ) + α₂·Σ h_jk(x_j, x_k).
- Intercept = beta
- offset = log(lambda)
- main effects = s_i  
- interactions = h_jk
- alpha_1 / alpha_2: weights
- Centring rule: shape function - sample mean of shape function 

## 3. Subnetworks
- Main effect net: layers (input -> hidden, leaky ReLU -> output, exp activation), glorot initialising:
- Pair net: layers (), glorot initialising 
- Categorical terms: one-hot encoding 

## 4. Penalties
- Smoothness: 
    grid of 1000 uniformly spaced points
    finite difference used 
    absolute vs squared (paper's choice)

- Marginal clarity:
    to avoid any confounding between main and interaction effects
    formula: mean of sᵢ·h_ij over observations
    computed per batch

## 5. Monotonicity
- monotonic lattice regression
- interpolation formula: 𝑚(𝑋𝑗, 𝑋𝑘) = (1 − 𝛾1) (1 − 𝛾2)𝜆𝑎𝑏 + 𝛾1 (1 − 𝛾2)𝜆(𝑎+1)𝑏 + (1 − 𝛾1)𝛾2𝜆𝑎(𝑏+1) + 𝛾1𝛾2𝜆(𝑎+1) (𝑏+1)
- Constraint set 𝜆(𝑎+1)𝑏 − 𝜆𝑎𝑏 ≥ 0, for all 𝑎 ∈ {0, 1, . . . , 𝑀𝑗 − 2}, 𝑏 ∈ {0, 1, . . . , 𝑀𝑘 − 1}.
- Projection: argmin distance between 𝝀𝑗 and 𝝀ˆj
- The algorithm works by iteratively projecting the estimate of 𝝀𝑗 onto each feasible set D𝑖 until convergence or the maximum number of iterations is attained

- synthetic data f34 shows X3 has a monotically decreasing relationship with X4

- bm is expected to have monotically increasing relationship with claim frequency 
-  For pairs involving the bm variable, a lattice model with 8 vertices along each dimension is applied to enforce the expected monotonic relationship. The input to this lattice model is scaled to the appropriate range using a piecewise linear calibrator with 20 knots, ensuring consistency in input scaling and enabling accurate function approximation.

## 6. Importance and selection
- Importance score formula: average of variance of shape function (step 1 main effects, step 2 interaction effects only if chosen in step 1)
- Stage 1: rank main effects by variance (descending), selecting top k1
- step 2: select top k2 interaction effects only if chosen in step 1
- 3 re-train all selected main and pairwise effects as a full model:

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
