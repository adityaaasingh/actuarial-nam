"""The deployable unit: raw policy rows in, expected claim counts out.

Bundles the fitted preprocessor with a fitted model, so the API never
re-implements preprocessing. Any model with .predict(preprocessed_df) fits here.
"""

import mlflow.pyfunc
import numpy as np
import pandas as pd


class PricingModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(model_input))
