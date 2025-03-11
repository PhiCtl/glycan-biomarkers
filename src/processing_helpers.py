from typing import List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import reduce


def load_data(paths: List[str], merge_on='sample', verbose=False):

    # Load and merge data
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if merge_on not in df.columns:
            raise ValueError(f"Merge column {merge_on} not found in {path}")
        dfs.append(df)
    data = reduce(lambda l, r : pd.merge(l, r, on=merge_on), dfs)
    if verbose:
        print(f"Successfully loaded {len(paths)} data files.")
    return data

def filter_types(data: pd.DataFrame, dict_args:dict, verbose: bool = False):

    for col, types in dict_args.items():
        data = data[data[col].isin(types)]
    if verbose: 
        print("Successfully filtered data.")
    return data.sort_values(by='order')


def normalize(data: pd.DataFrame, feat_cols: Optional[List[str]]=None, method: str='standard', verbose: bool = False):

    if feat_cols is None:
        feat_cols = [c for c in data.columns if c.startswith('FT')]

    if method == 'standard':
        scaler = StandardScaler()
        data[feat_cols] = scaler.fit_transform(data[feat_cols])
    else:
        raise ValueError(f"Normalization method {method} not found")
    if verbose: 
        print("Successfully normalized data.")
    return data

def pca(data: pd.DataFrame, group: str='class', feat_cols: Optional[List[str]]=None, verbose: bool = False):

    if feat_cols is None:
        feat_cols = [c for c in data.columns if c.startswith('FT')]
    
    assert(group in ['class', 'batch'])

    pca_2d = PCA(n_components=2)
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feat_cols])
    X_new = pca_2d.fit_transform(X)

    sns.set(rc = {'figure.figsize':(8, 5)})
    df_plot = pd.DataFrame(X_new, columns=['x1', 'x2'])
    df_plot[group] = data[group].values

    fig = sns.scatterplot(df_plot, x='x1', y='x2', hue=group)
    fig.set_title('PCA data projection on 2 main dimensions')
    plt.show(block=False)

    return data

def filter_features_detection(data: pd.DataFrame, feat_cols: Optional[List[str]] = None, qc_class: List[str] = ['QC'], threshold_value:int = 400, threshold_detect: float = 0.7, verbose: bool = False):

    if feat_cols is None:
        feat_cols = [c for c in data.columns if c.startswith('FT')]

    detection = data[data['class'].isin(qc_class)][feat_cols].apply(lambda x: x >= threshold_value, axis=0).astype(int)
    detected = detection.sum() / detection.shape[0]
    notdetected_features = list(detected[detected < threshold_detect].index)
    data_cols = set(data.columns) - set(notdetected_features)

    if verbose:
        print(f"Removing {len(notdetected_features)} features\
              with low detection rate:\n{notdetected_features}")

    return data[list(data_cols)]

def filter_features_variability(data: pd.DataFrame, feat_cols: Optional[List[str]] = None, qc_class: List[str] = ['QC'], threshold: float = 0.3, verbose: bool = False):

    if feat_cols is None:
        feat_cols = [c for c in data.columns if c.startswith('FT')]

    cv = data[data['class'].isin(qc_class)][feat_cols].apply(lambda x: x.std() / x.mean(), axis=0)
    high_var_features = list(cv[cv > threshold].index)
    data_cols = set(data.columns) - set(high_var_features)

    if verbose:
        print(f"Removing {len(high_var_features)} features\
              with high variability :\n{high_var_features}")

    return data[list(data_cols)]

def filter_features_dratio(data: pd.DataFrame, sample_classes: List[str], feat_cols: Optional[List[str]] = None,  qc_class: List[str] = ['QC'], threshold: float = 0.3, verbose: bool = False):

    if feat_cols is None:
        feat_cols = [c for c in data.columns if c.startswith('FT')]
    
    if not isinstance(qc_class, list):
        qc_class = [qc_class]

    qc_std = data.loc[data['class'].isin(qc_class)][feat_cols].std()
    sample_std = data.loc[data['class'].isin(sample_classes)][feat_cols].std()
    dratio = qc_std / (qc_std**2 + sample_std**2)

    low_biological_info_features = list(dratio[dratio > threshold].index)
    data_cols = set(data.columns) - set(low_biological_info_features)

    if verbose:
        print(f"Removing {len(low_biological_info_features)} features\
               with low biological information :\n{low_biological_info_features}")

    return data[list(data_cols)]

def filter_outliers(data, feat_cols, threshold=3):
    # TODO: Implement outlier detection
    pass

def split_train_test(data, test_size=0.2, random_state=42, verbose: bool = False):

    feat_cols = [c for c in data.columns if c.startswith('FT')]
    classes = data['class'].unique().tolist()
    X = data[feat_cols].to_numpy()
    y = data['class'].apply(lambda x : classes.index(x)).to_numpy()
    classes = { i : c for i, c in enumerate(classes)}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    if verbose:
        print("Successfully splitted train / test set.")

    return  X_train, X_test, y_train, y_test, classes, feat_cols

    