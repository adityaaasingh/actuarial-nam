import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from anam.metrics import evaluate_poisson, mae, poisson_deviance, poisson_nll, rmse


def test_nll_is_keras_poisson_loss():
    """Keras' loss = true Poisson NLL minus the log(y!) constant (up to its 1e-7 epsilon)."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.05, 2.0, 1000)
    y = rng.poisson(mu)
    true_nll = -poisson.logpmf(y, mu).mean()
    assert np.isclose(poisson_nll(y, mu), true_nll - gammaln(y + 1).mean(), rtol=1e-6)


def test_deviance_zero_at_perfect_fit_and_handles_zero_claims():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(poisson_deviance(y, y), 0.0)
    assert np.isclose(poisson_deviance([0.0], [0.4]), 0.8)  # 2 * mu when y = 0


def test_rmse_mae():
    assert np.isclose(rmse([0, 0], [3, 4]), np.sqrt(12.5))
    assert np.isclose(mae([0, 0], [3, 4]), 3.5)


def test_evaluate_returns_all_metrics():
    assert set(evaluate_poisson([0, 1], [0.5, 0.5])) == {
        "nll",
        "deviance",
        "rmse",
        "mae",
    }
