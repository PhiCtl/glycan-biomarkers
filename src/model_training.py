import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import random

random.seed(42)

class ModelTrainer:

    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.cv = 5
        self.best_model = None

    def tune_model(self, x, y, score='accuracy'):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.cv, scoring=score)
        grid_search.fit(x, y)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

    def train_model(self, x, y):
        if self.best_model is None:
            raise Exception("Model has not been tuned yet. Please run tune_model() first.")
        self.best_model.fit(x, y)
    
    def evaluate_model(self, x, y, metrics):
        predictions = self.best_model.predict(x)
        score = metrics(y, predictions)
        print(f"Model scoring: {score}")

def plot_feature_importance(model, features, top_n=10):
    """
    Plots the feature importance of a given model.
    Parameters:
    model : object
        The trained model with feature importances.
    features : list
        The list of feature names.
    top_n : int, optional
        The number of top features to display (default is 10).
    path : str, optional
        The directory path where the plot image will be saved. If None, the plot will not be saved (default is None).
    Returns:
    None
    """
    hasattr(model, "feature_importances_")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [features[i] for i in indices[:top_n]], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xlim([-1, top_n])
    plt.show(block=False)
    return [features[i] for i in indices[:top_n]]


class ShapWrapper:
    def __init__(self, model, X_train: pd.DataFrame, explainer_type: str = "auto"):
        """
        Wrapper for computing SHAP values.

        Parameters:
        - model: Trained model (supports sklearn, XGBoost, LightGBM, etc.).
        - X_train: Training dataset (used for the SHAP explainer).
        - explainer_type: Type of SHAP explainer ('tree', 'kernel', or 'auto').
        """
        self.model = model
        self.X_train = X_train
        self.explainer = self._initialize_explainer(explainer_type)

    def _initialize_explainer(self, explainer_type: str):
        """Initializes the appropriate SHAP explainer."""
        if explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        elif explainer_type == "kernel":
            return shap.KernelExplainer(self.model.predict, self.X_train.sample(100))  # Subsampling for efficiency
        else:  # Auto-detect
            try:
                return shap.Explainer(self.model, self.X_train)
            except:
                return shap.KernelExplainer(self.model.predict, self.X_train.sample(100))

    def compute_shap_values(self, X_test: pd.DataFrame):
        """Computes SHAP values for the given dataset."""
        return self.explainer.shap_values(X_test)

    def plot_summary(self, X_test: pd.DataFrame):
        """Plots a SHAP summary plot to show feature importance."""
        shap_values = self.compute_shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test)

    def plot_force(self, X_test: pd.DataFrame, instance_idx: int = 0):
        """Plots a SHAP force plot for a single instance."""
        shap_values = self.compute_shap_values(X_test)
        shap.initjs()
        return shap.force_plot(
            self.explainer.expected_value, 
            shap_values[instance_idx], 
            X_test.iloc[instance_idx]
        )

