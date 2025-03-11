import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from data_processing import DataProcessor

def hyperparameter_tuning(X, y, model, param_grid, cv=3, verbose=1):
    """
    Perform hyperparameter tuning using GridSearchCV.
    Parameters:
    X (array-like or sparse matrix): The input data to fit.
    y (array-like): The target variable to try to predict in the case of supervised learning.
    model (estimator object): The object to use to fit the data.
    param_grid (dict or list of dictionaries): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries.
    cv (int, cross-validation generator or an iterable, optional): Determines the cross-validation splitting strategy. Default is 3.
    verbose (int, optional): Controls the verbosity: the higher, the more messages. Default is 1.
    Returns:
    dict: The best parameters found during the grid search.
    """

    # Input validation
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("X should be a numpy array or a pandas DataFrame.")
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise ValueError("y should be a numpy array or a pandas Series.")
    if not hasattr(model, 'fit'):
        raise ValueError("model should be an estimator object with a fit method.")
    if not isinstance(param_grid, (dict, list)):
        raise ValueError("param_grid should be a dictionary or a list of dictionaries.")
    if not isinstance(cv, (int, StratifiedKFold)):
        raise ValueError("cv should be an integer or a StratifiedKFold object.")

    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='balanced_accuracy')
    grid_search.fit(X, y)

    if verbose:
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
    return grid_search.best_params_

def plot_feature_importance(model, features, top_n=10, path=None):
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

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), features[indices[:top_n]], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xlim([-1, top_n])

    if path:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, 'features_importance.png'))
    plt.show()






def main():
    
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    DATA_PATH = "../data/input"
    data_processor = DataProcessor(classes=['Dunn', 'French', 'LMU'])

    data_raw = data_processor.load_data(DATA_PATH)
    data_processed = data_processor.preprocess_data(data_raw)
    X, y = data_processor.prepare_for_training(data_processed)

    class_map = {'LMU' : 'Benign disease', 'French' : 'Lung cancer', 'Dunn' : 'Healthy'}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [5, 10, 15, 20, 25, 30],
        "max_depth": [3],
        "max_features":['sqrt'],
        "bootstrap": [True]
    }

    print("Hyper parameter tuning.")
    best_params = hyperparameter_tuning(X_train, y_train, RandomForestClassifier(random_state=42), param_grid)

    print("Model training.")
    model = RandomForestClassifier(**best_params, random_state=42, oob_score=True)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print('Classification report :\n', classification_report(y_test, y_pred,\
                                                            target_names=[class_map[k] for k in data_processor.classes]))
    print('Confusion matrix :\n', confusion_matrix(y_test, y_pred))

    # Plot feature importance
    plot_feature_importance(model, data_processed['feature'].unique(), top_n=10, path="../outputs")


if __name__ == '__main__':

    main()