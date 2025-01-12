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

def scale_intensities(df_data, scaling_mode='median'):

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
    else:
        return df_data

def select_features(df_data, CV_thresh=30, detect_thresh=400, detect_perc=70):

    # Feature selection with CV < 30% for QC
    # Discard outliers features in QC samples

    # Remove features with outliers in QC to compute robust CV on features
    df_QC = df_data[df_data['class'] == 'QC']
    df_QC_summary = df_QC.groupby(['feature'])['intensity'].describe().reset_index()
    df_QC_summary['lower_bound'] = 2.5*df_QC_summary['25%'] - 1.5*df_QC_summary['75%']
    df_QC_summary['upper_bound'] = 2.5*df_QC_summary['75%'] - 1.5*df_QC_summary['25%']
    df_QC = df_QC.merge(df_QC_summary[['feature', 'lower_bound', 'upper_bound']], on='feature')

    # Since the features first selection step relies on CV per feature, we can just remove feature with outlier value for each sample
    df_QC = df_QC[(df_QC['intensity'] >= df_QC['lower_bound']) & (df_QC['intensity'] <= df_QC['upper_bound']) ]
    # Compute feature CV
    cv_QC = df_QC.groupby('feature')['intensity'].agg(lambda x : x.std() / x.mean() * 100).reset_index().rename(columns={'intensity' : 'CV%'})
    selected_feats = list(cv_QC[cv_QC['CV%'] <= CV_thresh]['feature'])
    df_data = df_data[df_data['feature'].isin(selected_feats)]

    # Feature consistency in detection (>= 70%) 
    df_data['detection'] = df_data['intensity'].apply(lambda x: x > detect_thresh)
    df_detect = df_data.groupby('feature').agg({'detection':lambda x : x.sum() / x.size * 100})
    detected_features = df_detect.loc[(df_detect['detection'] < detect_perc)].index
    print(f"At this stage, {len(detected_features)} features were selected")
    df_data = df_data[df_data['feature'].isin(detected_features)]

    return df_data

def remove_outliers(df_data, out_thresh=4, in_classes=['Dunn', 'French', 'LMU']):


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

    return df_data.drop(['lower_bound', 'upper_bound'], axis=1)



class DataProcessor:

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
    
    def process_data(self, df_data):

        #Filter glycans
        df_data = df_data[df_data['mz'] > 500].drop(['mz', 'rt'], axis=1)

        # Filter classes & batch
        df_data = df_data[~df_data['class'].isin(['B', 'SS', 'dQC']) & (df_data['batch'] == 1)]

        df_data = select_features(df_data)
        df_data = remove_outliers(df_data)
        df_data = scale_intensities(df_data)
        

        return df_data

    def save_data(self, df_data, path):
        df_data.to_csv(path)


def main():

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    DATA_PATH = "../../data/input"
    data_processor = DataProcessor(classes=['Dunn', 'French', 'LMU'])

    data_raw = data_processor.load_data(DATA_PATH)
    data_processed = data_processor.process_data(data_raw)

    print(f"Initial list contains {data_raw['feature'].nunique()} features and {data_raw[data_raw['class'].isin(['Dunn', 'French', 'LMU'])]['sample'].nunique()} samples.")
    print(f"Final curated list contains {data_processed['feature'].nunique()} features and {data_processed[data_processed['class'].isin(['Dunn', 'French', 'LMU'])]['sample'].nunique()} samples.")


if __name__ == '__main__':
    main()