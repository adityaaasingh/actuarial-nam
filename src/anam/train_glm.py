"""Fit the Poisson GLM on beMTPL97, log it to MLflow and register it as `champion`.

Run after `make data`. Usage: uv run python -m anam.train_glm [--encoding ordinal|onehot]
"""

import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

from anam.baselines.glm import PoissonGLM
from anam.data.bemtpl import (
    CATEGORICAL,
    CONTINUOUS,
    EXPOSURE,
    FEATURES,
    PROCESSED_PATH,
    TARGET,
)
from anam.data.prepare import ARTIFACTS
from anam.data.preprocess import Preprocessor
from anam.data.splits import load_split
from anam.metrics import evaluate_poisson
from anam.pricing_model import PricingModel

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "bemtpl-frequency"
MODEL_NAME = "anam-pricing"
PAPER_GLM = {"nll": 0.3803, "rmse": 0.3686, "mae": 0.2177}  # paper's GLM row, test set


def main(encoding: str, register: bool) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_parquet(PROCESSED_PATH)
    splits = load_split(df, ARTIFACTS / "bemtpl_splits.parquet")
    pre = Preprocessor.load(ARTIFACTS / "bemtpl_preprocessor.joblib")
    X = {k: pre.transform(v) for k, v in splits.items()}

    glm = PoissonGLM(CATEGORICAL, CONTINUOUS, pre.levels_, encoding=encoding)
    glm.fit(X["train"], splits["train"][TARGET])

    with mlflow.start_run(run_name=f"glm-{encoding}") as run:
        mlflow.set_tags({"model": "glm", "dataset": "bemtpl97"})
        mlflow.log_params(
            {"model": "poisson_glm", "encoding": encoding, "n_coef": len(glm.coef_)}
        )
        for split_name in ("val", "test"):
            metrics = evaluate_poisson(
                splits[split_name][TARGET], glm.predict(X[split_name])
            )
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items()})
            print(split_name, {k: round(v, 5) for k, v in metrics.items()})
        mlflow.log_dict(glm.relativities().round(4).to_dict(), "relativities.json")

        example = (
            splits["train"][[EXPOSURE, *FEATURES]]
            .head(3)
            .astype({c: str for c in CATEGORICAL})
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PricingModel(pre, glm),
            input_example=example,
            registered_model_name=MODEL_NAME if register else None,
        )
        print(f"paper GLM (test): {PAPER_GLM}")
        print(f"run {run.info.run_id}")

    if register:
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
        print(f"registered {MODEL_NAME} v{version} as champion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", choices=["ordinal", "onehot"], default="onehot")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    main(args.encoding, register=not args.no_register)
