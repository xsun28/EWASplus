#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:00:11 2017

@author: Xiaobo
"""

from statsmodels.stats.weightstats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureTTest(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        self.col = col
        
    def fit(self,pos,neg):
        tstat,pvalue,df = ttest_ind(pos[self.col],neg[self.col])
        diff = np.mean(pos[self.col])-np.mean(neg[self.col])
        return {'feature':self.col,'tstats':tstat,'pvalue':pvalue,'diff(pos-neg)':diff}
    
    