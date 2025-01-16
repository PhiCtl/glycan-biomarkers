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
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='balanced_accuracy')
    grid_search.fit(X, y)

    if verbose:
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
    return grid_search.best_params_

def plot_feature_importance(model, features, top_n=10, path=None):
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