"""Evaluation metrics, defined to match the paper's tables.

The paper's "NLL" for Poisson is Keras' `poisson` loss: mean(mu - y * log(mu + 1e-7)).
It drops the log(y!) constant, so it is not the true negative log-likelihood;
`poisson_nll` reproduces it exactly so our numbers are comparable with the paper.
"""

import numpy as np

KERAS_EPS = 1e-7


def poisson_nll(y, mu) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(np.mean(mu - y * np.log(mu + KERAS_EPS)))


def poisson_deviance(y, mu) -> float:
    """Mean Poisson unit deviance: 2 * [y log(y/mu) - (y - mu)], with 0 log 0 = 0."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (ylogy - (y - mu))))


def rmse(y, mu) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(mu, float)) ** 2)))


def mae(y, mu) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(mu, float))))


def evaluate_poisson(y, mu) -> dict[str, float]:
    return {
        "nll": poisson_nll(y, mu),
        "deviance": poisson_deviance(y, mu),
        "rmse": rmse(y, mu),
        "mae": mae(y, mu),
    }
