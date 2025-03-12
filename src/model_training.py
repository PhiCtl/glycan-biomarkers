import random
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED) 

class ModelTrainer:

    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.cv = 5
        self.best_model = None
        self.seed = SEED

    def tune_model(self, x, y, score='accuracy'):
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid,\
                                        cv=cv, scoring=score)
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
    def __init__(self, model, background: pd.DataFrame, classes: dict, explainer_type: str = "auto"):
        """
        Wrapper for computing SHAP values.

        Parameters:
        - model: Trained model (supports sklearn, XGBoost, LightGBM, etc.).
        - X_train: Training dataset (used for the SHAP explainer).
        - explainer_type: Type of SHAP explainer ('tree', 'kernel', or 'auto').
        """
        self.model = model
        self.background = background
        self.classes = classes
        self.explainer = self._initialize_explainer(explainer_type)

    def _initialize_explainer(self, explainer_type: str):
        """Initializes the appropriate SHAP explainer."""
        if explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        elif explainer_type == "kernel":
            return shap.KernelExplainer(self.model.predict, self.background.sample(frac=.7))  # Subsampling for efficiency
        else:  # Auto-detect
            try:
                return shap.Explainer(self.model, self.background.sample(frac=.7))
            except:
                return shap.KernelExplainer(self.model.predict, self.background.sample(frac=.7))

    def compute_shap_values(self, X_test: pd.DataFrame):
        """Computes SHAP values for the given dataset."""
        return self.explainer(X_test)

    def plot_summary(self, X_test: pd.DataFrame, cl: str, plot_type: str = "beeswarm", sample_ind: int = None):
        """Plots a SHAP summary plot to show feature importance."""
        assert(cl in self.classes.keys())
        assert(plot_type in ["beeswarm", "force", "bar", "waterfall"])
        if plot_type == "waterfall":
            assert(sample_ind is not None)

        shap_values = self.compute_shap_values(X_test)[:,:,self.classes[cl]]    

        plt.figure(figsize=(10, 6))
        if plot_type == "beeswarm":
            shap.plots.beeswarm(shap_values)
        elif plot_type == "bar":
            shap.plots.bar(shap_values)
        elif plot_type == "waterfall":
            shap.plots.waterfall(shap_values[sample_ind])
        elif plot_type == "force":
            shap.plots.force(shap_values)
        plt.show(block=False)

