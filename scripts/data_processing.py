import os
import sys
from pathlib import Path
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scipy.stats import median_abs_deviation as mad

def d_ratio(df_data, sample_classes, qc_classes):

    """
    Calculate the D-ratio and standard deviation ratio for features in the dataset.

    Parameters:
    df_data (pd.DataFrame): DataFrame containing the data with columns 'class', 'batch', 'feature', and 'intensity'.
    sample_classes (list): List of sample class names to include in the calculation.
    qc_classes (list): List of QC class names to include in the calculation.

    Returns:
    pd.DataFrame: DataFrame with columns 'feature', 'std_sample', 'std_QC', 'D_ratio', and 'std_ratio'.
    """

    if not isinstance(df_data, pd.DataFrame):
        raise ValueError("df_data must be a pandas DataFrame")
    if not isinstance(sample_classes, list) or not all(isinstance(i, str) for i in sample_classes):
        raise ValueError("sample_classes must be a list of strings")
    if not isinstance(qc_classes, list) or not all(isinstance(i, str) for i in qc_classes):
        raise ValueError("qc_classes must be a list of strings")

    QC_std = df_data[df_data['class'].isin(qc_classes) & (df_data['batch'] == 1)]\
            .groupby(['feature']).agg({'intensity':lambda x : mad(x)})\
            .rename(columns={'intensity' : 'std_QC'})
    
    general_std = df_data[df_data['class'].isin(sample_classes)]\
            .groupby(['feature']).agg({'intensity':lambda x : mad(x) })\
            .rename(columns={'intensity' : 'std_sample'})

    d_ratio = general_std.reset_index().merge(QC_std.reset_index(), on='feature', how='outer')
    d_ratio['D_ratio'] = d_ratio.apply(lambda x : 100 * x.std_QC / (np.sqrt(x.std_QC**2 + x.std_sample**2)), axis=1)
    d_ratio['std_ratio'] = d_ratio['std_sample'] / d_ratio['std_QC']

    return d_ratio

def subbatch(x):
    if x['order'] <= 14 :
        return 0
    if x['order'] <= 26 :
        return 1
    if x['order'] <= 38 :
        return 2
    if x['order'] <= 50 :
        return 3
    if x['order'] <= 62 :
        return 4
    if x['order'] <= 74 :
        return 5
    if x['order'] <= 86 :
        return 6
    if x['order'] <= 101 :
        return 7
    else:
        return 8

def scale_intensities(df_data, scaling_mode='log'):
    """
    Scale the intensities in the dataset using the specified scaling mode.

    Parameters:
    df_data (pd.DataFrame): DataFrame containing the data with columns 'class', 'batch', 'feature', 'intensity', etc.
    scaling_mode (str): The scaling mode to use. Options are 'median' for median scaling and 'log' for log transformation.

    Returns:
    pd.DataFrame: DataFrame with scaled intensities.
    """

    assert(scaling_mode in ['median', 'log'])

    if scaling_mode == 'median':
        # Compute the QC pool metabolites median intensities for each data sub_batch
        pools_medians = df_data[df_data['class'] == 'QC'].groupby(['sub_batch', 'feature'])['intensity'].median().reset_index()
        pools_medians = pools_medians.sort_values(['feature', 'sub_batch'])\
                                    .groupby('feature')\
                                    .rolling(3, min_periods=2, center=True).median()\
                                    .replace({'sub_batch':6.5}, 7).replace({'sub_batch':0.5}, 0)\
                                    .reset_index('feature').rename(columns={'intensity' : 'med_QC_intensity'})
        df_data_corr = df_data.merge(pools_medians, on=['feature', 'sub_batch'], how='outer')
        df_data_corr['intensity_raw'] = df_data_corr['intensity']
        df_data_corr['intensity'] = df_data_corr['intensity'] / df_data_corr['med_QC_intensity']

        return df_data_corr.drop(['order', 'sub_batch', 'batch', 'med_QC_intensity'], axis=1)
    
    elif scaling_mode == 'log':
        # Log transformation of the intensities
        df_data['log_intensity'] = df_data['intensity'].apply(lambda x : np.nan if x <= 1E-3 else np.log(x))
        df_data['log_intensity'] = df_data[['feature', 'log_intensity']].groupby('feature').transform(lambda x : x.fillna(x.median()))
        df_data['intensity_raw'] = df_data['intensity']
        df_data['intensity'] = df_data['log_intensity']
        return df_data.drop(['log_intensity'], axis=1)

def select_features(df_data, CV_thresh=30, detect_thresh=500, detect_perc=70, d_ratio=False):

    """
    Select features based on coefficient of variation (CV), detection threshold, and optionally D-ratio.

    Parameters:
    df_data (pd.DataFrame): DataFrame containing the data with columns 'class', 'batch', 'feature', 'intensity', etc.
    CV_thresh (float): Threshold for the coefficient of variation (CV) in QC samples.
    detect_thresh (float): Intensity threshold for feature detection.
    detect_perc (float): Minimum percentage of samples in which a feature must be detected.
    d_ratio (bool): Whether to apply D-ratio filtering.

    Returns:
    pd.DataFrame: DataFrame with selected features.
    """

    if not isinstance(df_data, pd.DataFrame):
        raise ValueError("df_data must be a pandas DataFrame")
    if not isinstance(CV_thresh, (int, float)):
        raise ValueError("CV_thresh must be a numeric value")
    if not isinstance(detect_thresh, (int, float)):
        raise ValueError("detect_thresh must be a numeric value")
    if not isinstance(detect_perc, (int, float)):
        raise ValueError("detect_perc must be a numeric value")
    if not isinstance(d_ratio, bool):
        raise ValueError("d_ratio must be a boolean value")

    # Feature selection with CV < 30% for QC and consistency in feature detection >= 70%
    df_QC = df_data[df_data['class'] == 'QC']

    # Compute feature CV with MAD approx
    cv_QC = df_QC.groupby('feature')['intensity'].agg(lambda x : 1.4826 * mad(x) / x.median() * 100)\
        .reset_index().rename(columns={'intensity' : 'CV%'})
    selected_feats = list(cv_QC[cv_QC['CV%'] <= CV_thresh]['feature'])
    df_data = df_data[df_data['feature'].isin(selected_feats)]

    print(f"{len(selected_feats)} features have a CV of at most {CV_thresh}%.")

    # Feature consistency in detection (>= 70%) 
    df_data['detection'] = df_data['intensity'].apply(lambda x: x >= detect_thresh)
    df_detect = df_data.groupby('feature').agg({'detection':lambda x : x.sum() / x.size * 100})
    detected_features = df_detect.loc[(df_detect['detection'] >= detect_perc)].index
    df_data = df_data[df_data['feature'].isin(detected_features)]
    print(f"{len(detected_features)} features detected in at least {detect_perc}% of samples.")

    # Select features with d-ratio < 50% if specified
    if d_ratio:

        df_d_ratio = d_ratio(df_data, ['Dunn', 'French', 'LMU'], ['QC'])
        selected_feats_dratio = df_d_ratio[df_d_ratio['D_ratio'] <= 50]['feature'].to_list()
        df_data = df_data[df_data['feature'].isin(selected_feats_dratio)]

    return df_data

def remove_outliers(df_data, out_thresh=4, in_classes=['Dunn', 'French', 'LMU']):

    """
    Remove outliers from the dataset based on intensity values.

    Parameters:
    df_data (pd.DataFrame): DataFrame containing the data with columns 'class', 'sample', 'feature', 'intensity', etc.
    out_thresh (int): Threshold for the number of outliers to exclude a sample.
    in_classes (list): List of class names to include in the outlier removal process.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """

    if not isinstance(df_data, pd.DataFrame):
        raise ValueError("df_data must be a pandas DataFrame")
    if not isinstance(out_thresh, int):
        raise ValueError("out_thresh must be an integer")
    if not isinstance(in_classes, list) or not all(isinstance(i, str) for i in in_classes):
        raise ValueError("in_classes must be a list of strings")


    summ = df_data.groupby('feature')['intensity'].describe().reset_index()
    summ['lower_bound'] = 2.5*summ['25%'] - 1.5*summ['75%']
    summ['upper_bound'] = 2.5*summ['75%'] - 1.5*summ['25%']
    df_data = df_data.merge(summ[['feature', 'lower_bound', 'upper_bound']], on='feature')

    # Detect oultiers
    df_outliers = df_data[(df_data['intensity'] < df_data['lower_bound']) | (df_data['intensity'] > df_data['upper_bound']) ]\
                .groupby(['class','sample']).size().reset_index()
    df_outliers = df_outliers.rename(columns={0:'df_outliers'})
    
    # Exclude
    samples_to_exclude = list(df_outliers[(df_outliers['df_outliers'] > out_thresh) & df_outliers['class'].isin(in_classes)]['sample'])
    df_data = df_data[~df_data['sample'].isin(samples_to_exclude)]
    print(f"{len(samples_to_exclude)} samples have been excluded due to outliers.")

    return df_data.drop(['lower_bound', 'upper_bound'], axis=1)



class DataProcessor:

    """
    DataProcessor class for processing and preparing data for training.

    Attributes:
        classes (list): List of class names to include in the processing.
        test_size (float): Proportion of the dataset to include in the test split.
        rd (int): Random seed for reproducibility.

    Methods:
        __init__(self, classes, test_size=.2, rd_seed=42):
            Initializes the DataProcessor with the specified classes, test size, and random seed.

        load_data(self, path):
            Loads the data from the specified path and augments it with metadata.

        preprocess_data(self, df_data):
            Preprocesses the data by filtering glycans, classes, and batch, selecting features, removing outliers, and scaling intensities.

        prepare_for_training(self, df_data):
            Prepares the data for training by pivoting the DataFrame and converting it to numpy arrays for features and labels.

        save_data(self, df_data, path):
            Saves the processed data to the specified path.
    """

    def __init__(self,
                 classes,
                 test_size=.2,
                 rd_seed=42):
        
        self.classes = classes
        self.test_size = test_size
        self.rd = rd_seed
    
    def load_data(self,
                  path):
        
        df_data = pd.read_csv(os.path.join(path, 'internship_data_matrix.csv'))
        df_feature_meta = pd.read_csv(os.path.join(path, 'internship_feature_metadata.csv'))
        df_acq = pd.read_csv(os.path.join(path, 'internship_acquisition_list.csv'))

        # Augment dataset with metadata

        df_acq['sub_batch'] = df_acq.apply(lambda x : subbatch(x), axis=1)
        df_data = df_data.merge(df_acq[['sample', 'class', 'order', 'batch', 'sub_batch']], on='sample')
        df_data = df_data.melt(id_vars=['sample', 'class', 'order', 'batch', 'sub_batch'],value_name='intensity', var_name='feature')
        df_data = df_data.merge(df_feature_meta[['feature', 'mz', 'rt']], on='feature')
        df_data['feature_number'] = df_data['feature'].apply(lambda s : int(s.split('-')[1]))
        
        return df_data
    
    def preprocess_data(self, df_data):

        #Filter glycans
        df_data = df_data[df_data['mz'] > 500].drop(['mz', 'rt'], axis=1)

        # Filter classes & batch
        df_data = df_data[~df_data['class'].isin(['B', 'SS', 'dQC']) & (df_data['batch'] == 1)]

        df_data = select_features(df_data)
        df_data = remove_outliers(df_data)
        df_data = scale_intensities(df_data)
        

        return df_data[df_data['class'].isin(self.classes)]
    
    def prepare_for_training(self, df_data):
        df_X = df_data[['sample','feature', 'intensity', 'class']].pivot(index=['sample', 'class'], columns='feature', values='intensity').reset_index()
        X = df_X.drop(['sample', 'class'], axis=1).to_numpy()
        y = df_X['class'].apply(lambda x : self.classes.index(x)).to_numpy()

        return X, y

    def save_data(self, df_data, path):
        df_data.to_csv(path)


def main():

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    DATA_PATH = "../data/input"
    data_processor = DataProcessor(classes=['Dunn', 'French', 'LMU'])

    data_raw = data_processor.load_data(DATA_PATH)
    data_processed = data_processor.preprocess_data(data_raw)

    print(f"Initial list contains {data_raw['feature'].nunique()} features and {data_raw[data_raw['class'].isin(['Dunn', 'French', 'LMU'])]['sample'].nunique()} samples.")
    print(f"Final curated list contains {data_processed['feature'].nunique()} features and {data_processed['sample'].nunique()} samples.")


if __name__ == '__main__':
    main()