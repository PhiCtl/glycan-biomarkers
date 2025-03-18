# Standard library imports
from functools import reduce
from typing import List, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(paths, **kwargs):

    merge_on = kwargs.get("merge_on", None)
    verbose = kwargs.get("verbose", False)

    # Load and merge data
    dfs = []

    for path in paths:
        df = pd.read_csv(path)
        if merge_on is not None and merge_on not in df.columns:
            raise ValueError(f"Merge column {merge_on} not found in {path}")
        dfs.append(df)

    if verbose:
        print(f"Successfully loaded {len(paths)} data files.")

    if merge_on :
        data = reduce(lambda l, r : pd.merge(l, r, on=merge_on), dfs)
        return data

    return dfs

def filter_data(data: pd.DataFrame, **kwargs):

    types = kwargs.get("types", {})
    verbose = kwargs.get("verbose", False)

    if not isinstance(types, dict):
        raise TypeError("The 'types' parameter must be a dictionary.")

    for col, values in types.items():
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        if not isinstance(values, list):
            raise TypeError(f"Values for column '{col}' must be provided as a list.")
        data = data[data[col].isin(values)]

    if verbose: 
        print("Successfully filtered data.")

    if 'order' not in data.columns:
        raise ValueError("Column 'order' not found in the DataFrame for sorting.")

    return data.sort_values(by='order', ascending=True)


def normalize(data: pd.DataFrame, **kwargs):

    feat_cols = kwargs.get("feat_cols", [c for c in data.columns if c.startswith('FT')])
    verbose = kwargs.get("verbose", False)

    scaler = StandardScaler()
    data[feat_cols] = scaler.fit_transform(data[feat_cols])
    if verbose: 
        print("Successfully normalized data.")
    return data

def pca(data: pd.DataFrame, **kwargs):

    group = kwargs.get('group','class')
    feat_cols = kwargs.get('feat_cols', [c for c in data.columns if c.startswith('FT')]) 
    standardize = kwargs.get('standardize', False)
    assert(group in ['class', 'batch'])

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(data[feat_cols])
    else:
        X = data[feat_cols]

    pca_2d = PCA(n_components=2)
    X_new = pca_2d.fit_transform(X)

    sns.set_theme(rc = {'figure.figsize':(8, 5)})
    df_plot = pd.DataFrame(X_new, columns=['x1', 'x2'])
    df_plot[group] = data[group].values

    fig = sns.scatterplot(df_plot, x='x1', y='x2', hue=group)
    fig.set_title('PCA data projection on 2 main dimensions')
    plt.show(block=False)

    return data

def filter_features_detection(data: pd.DataFrame, **kwargs):
    feat_cols = kwargs.get("feat_cols", [c for c in data.columns if c.startswith('FT')])
    qc_class = kwargs.get("qc_class", ['QC'])
    threshold_value = kwargs.get("threshold_value", 500)
    threshold_detect = kwargs.get("threshold_detect", 0.7)
    verbose = kwargs.get("verbose", False)

    detection = data[data['class'].isin(qc_class)][feat_cols].apply(lambda x: x >= threshold_value, axis=0).astype(int)
    detected = detection.sum() / detection.shape[0]
    notdetected_features = list(detected[detected < threshold_detect].index)
    data_cols = set(data.columns) - set(notdetected_features)

    if verbose:
        print(f"Removing {len(notdetected_features)} features\
              with low detection rate:\n{notdetected_features}")

    return data[list(data_cols)]

def filter_features_variability(data: pd.DataFrame, **kwargs):
    feat_cols = kwargs.get("feat_cols", [c for c in data.columns if c.startswith('FT')])
    qc_class = kwargs.get("qc_class", ['QC'])
    threshold = kwargs.get("threshold", 0.3)
    verbose = kwargs.get("verbose", False)

    cv = data[data['class'].isin(qc_class)][feat_cols].apply(lambda x: x.std() / x.mean(), axis=0)
    high_var_features = list(cv[cv > threshold].index)
    data_cols = set(data.columns) - set(high_var_features)

    if verbose:
        print(f"Removing {len(high_var_features)} features\
              with high variability :\n{high_var_features}")

    return data[list(data_cols)]

def filter_features_dratio(data: pd.DataFrame, **kwargs):

    feat_cols = kwargs.get("feat_cols", [c for c in data.columns if c.startswith('FT')])
    sample_classes = kwargs.get("sample_classes")
    assert sample_classes is not None

    qc_classes = kwargs.get("qc_classes", ['QC'])
    threshold = kwargs.get("threshold", 0.5)
    verbose = kwargs.get("verbose", False)

    qc_std = data.loc[data['class'].isin(qc_classes)][feat_cols].std()
    sample_std = data.loc[data['class'].isin(sample_classes)][feat_cols].std()
    dratio = qc_std / (qc_std**2 + sample_std**2)**.5

    low_biological_info_features = list(dratio[dratio > threshold].index)
    data_cols = set(data.columns) - set(low_biological_info_features)

    if verbose:
        print(f"Removing {len(low_biological_info_features)} features\
               with low biological information :\n{low_biological_info_features}")

    return data[list(data_cols)]

def filter_outliers(data, feat_cols, threshold=3):
    # TODO: Implement outlier detection
    pass

def split_train_test(data, **kwargs): 

    test_size = kwargs.get("test_size", 0.2)
    verbose = kwargs.get("verbose", False)
    feat_cols = sorted([c for c in data.columns if c.startswith('FT')])
    seed = kwargs.get("seed", 42)

    classes = data['class'].unique().tolist()
    X = data[feat_cols].to_numpy()
    y = data['class'].apply(lambda x : classes.index(x)).to_numpy()
    classes = { i : c for i, c in enumerate(classes)}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                         stratify=y)

    if verbose:
        print("Successfully splitted train / test set.")

    return  X_train, X_test, y_train, y_test, classes, feat_cols

    