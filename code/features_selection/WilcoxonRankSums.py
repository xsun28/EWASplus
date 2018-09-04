#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:00:11 2017

@author: Xiaobo
"""

from scipy.stats import ranksums
from sklearn.base import BaseEstimator
import numpy as np

class Ranksums(BaseEstimator):
    
    def __init__(self,col):
        self.col = col
        
    def fit(self,pos,neg):
        stat,pvalue = ranksums(pos[self.col],neg[self.col])
        diff = np.mean(pos[self.col])-np.mean(neg[self.col])
        return {'feature':self.col,'stats':stat,'pvalue':pvalue,'diff(pos-neg)':diff}