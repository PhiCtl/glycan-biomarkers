from sklearn.exceptions import NotFittedError
from src.task_1.processing_pipeline import PipelineStep
from src.task_1.processing_helpers import pca

from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd

"""
Adapted from tidyms : https://github.com/griquelme/tidyms/blob/master/src/tidyms/_batch_corrector.py
"""


class _LoessCorrector(BaseEstimator, RegressorMixin):
    """
    Intra-batch corrector implementation using sklearn. From tidyms.

    Parameters
    ----------
    frac : number between 0 and 1, default=0.66
        Fraction of samples used for local regressions

    """

    def __init__(self, frac: float = 0.66):
        """
        Constructor function.

        """

        self.frac = frac
        self.interpolator_ = None

    def fit(self, X, y):
        x = X.flatten()
        y_fit = lowess(y, x, frac=self.frac, is_sorted=True, return_sorted=False)
        fill = (y[0], y[-1])
        self.interpolator_ = interp1d(x, y_fit, fill_value=fill, bounds_error=False)
        return self

    def predict(self, X):
        if self.interpolator_ is None:
            raise NotFittedError
        xf = X.flatten()
        x_interp = self.interpolator_(xf)
        return x_interp



class BatchCorrector(PipelineStep):

    # TODO cross validated frac_loess parameter

    def __init__(self, name='Batch Corrector', func=None):
        super().__init__(name=name)
        self.feature_thresh = 400
        self.min_detection_rate = 0.9
        self.frac_loess = 0.66
        self.qc_classes = ['QC']
        self.sample_classes = []


    def remove_invalid_features(self, data, feat_cols, verbose=False):

        """ Remove features with low detection rate"""

        invalid_features = set()

        for _, df in data.groupby('batch'):
            qc_mask = df['class'].isin(['QC'])
            sample_mask = df['class'].isin(['Healthy control', 'Cancer', 'Benign disease'])
            sample_order_min = df.loc[sample_mask, 'order'].min()
            sample_order_max = df.loc[sample_mask, 'order'].max()

            first_qc_block = qc_mask & (df['order'] > sample_order_min)
            first_qc_block_detection_rate = (df.loc[first_qc_block, feat_cols] > self.feature_thresh).sum(axis=0) / first_qc_block.sum()
            feat_first_qc = list((first_qc_block_detection_rate[first_qc_block_detection_rate < self.min_detection_rate]).index)

            last_qc_block = qc_mask & (df['order'] < sample_order_max)
            last_qc_block_detection_rate = (df.loc[last_qc_block, feat_cols] > self.feature_thresh).sum(axis=0) / last_qc_block.sum()
            feat_last_qc = list((last_qc_block_detection_rate[last_qc_block_detection_rate < self.min_detection_rate]).index)

            middle_qc_block = qc_mask & (df['order'] > sample_order_min) & (df['order'] < sample_order_max)
            middle_qc_block_detection_rate = (df.loc[middle_qc_block, feat_cols] > self.feature_thresh).sum(axis=0) / middle_qc_block.sum()
            feat_middle_qc = list((middle_qc_block_detection_rate[middle_qc_block_detection_rate < self.min_detection_rate]).index)

            invalid_features.update(set(feat_first_qc).union(feat_last_qc).union(feat_middle_qc))
        if verbose: print(f"Removing {len(invalid_features)} features:\n{invalid_features}")

        data = data.drop(columns=invalid_features, axis=1)
        return data, [c for c in data.columns if c.startswith('FT') ]

        
    def remove_invalid_samples(self, data, verbose=False):

        """ Remove samples not surrounded by QC samples"""

        invalid_samples = []

        for _, df in data.groupby('batch'):

            qc_order = df.loc[df['class'].isin(self.qc_classes), 'order']
            samples_order = df.loc[df['class'].isin(self.sample_classes), 'order']
            out_of_order_mask = (samples_order < qc_order.min()) | (samples_order > qc_order.max())

            invalid_samples.extend(df.loc[df['class'].isin(self.sample_classes), 'sample'][out_of_order_mask].values)
        
        if verbose: print(f"Removing {len(invalid_samples)} samples:\n{invalid_samples}")

        return data.loc[~data['sample'].isin(invalid_samples)]
    
    def correct_intrabatch_effects(self, data, feat_cols):

        # TODO implement per batch and parallelize and determine optimal fraction of samples to use
        data = data.sort_values(by='order')
        correction_mask = data['class'].isin(self.qc_classes)
        to_correct_mask = data['class'].isin(self.sample_classes + self.qc_classes)
        x_train = data.loc[correction_mask, 'order'].values
        x_pred = data.loc[to_correct_mask, 'order'].values 
        

        corrector = _LoessCorrector(frac=self.frac_loess)

        for f in feat_cols:

            # median feature value
            mean_f = data.loc[data['class'].isin(self.qc_classes), f].median()
            
            y_train = data.loc[correction_mask, f].values
            y_pred = data.loc[to_correct_mask, f].values

            # fit corrector on qc samples
            corrector.fit(x_train, y_train)
            # predict on all samples
            pred = corrector.predict(x_pred)
            factor = np.zeros_like(pred)
            # correct nan in zero values and negative values generated during LOESS
            is_positive = pred > 0
            factor = np.divide(mean_f, pred, out=factor, where=is_positive)
            corrected = y_pred * factor
            data.loc[to_correct_mask, f] = corrected

        return data
    
    def run(self, data, verbose=False):

        if verbose:
            print(f"Running {self.name}")

        feat_cols = [c for c in data.columns if c.startswith('FT')]
        #data, feat_cols = self.remove_invalid_features(data, feat_cols, verbose)
        data = self.remove_invalid_samples(data, verbose)
        data = self.correct_intrabatch_effects(data, feat_cols)
        return data