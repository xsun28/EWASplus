#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:00:56 2018

@author: Xiaobo
"""
import sys
home = '/home/ec2-user/git/EnsembleCpG/'
extra_storage = '/home/ec2-user/extra_storage/CpG_EWAS/'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from common import DataScaler as ds
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
###########################################
def sample_weights(X,y,class_weights=None,factor=0.5):
    weights = pd.Series(np.ones(X.shape[0]),index=y.index)
    if class_weights is not None:
        for cls,weight in class_weights.items():
            weights[y==cls] = weight
    else:
        controls_ix = y[y==0].index
        if X.ix[controls_ix,'pvalue'].isnull().sum() > 0:
            weights_list = [np.power(-(np.log(pvalue)),factor) if not math.isnan(pvalue) else 1 for pvalue in X['pvalue'] ]
            weights = pd.Series(weights_list,index=y.index)
        else:       
            weights = np.power(-(np.log(X['pvalue'])),factor)       
    return weights

#-----------------------------------------------------------------------------
def train_test_split(data,test_size=0.1,scaler='standard'):
    total_dataset = data.copy()
    total_dataset = total_dataset.reset_index().drop('index',axis=1)   #reset index or split below will generate filtered index and NAN values
    split = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=17)
    for train_index, test_index in split.split(total_dataset,total_dataset['label']):
        train_set = total_dataset.ix[train_index]
        test_set = total_dataset.ix[test_index]
    scaler = ds.DataScaler(scaler=scaler)
    train_x = scaler.fit_transform(train_set[train_set.columns.drop(['label','pvalue'])])
    train_x['pvalue'] = train_set['pvalue']
    train_label = train_set['label']
    test_x = scaler.transform(test_set[test_set.columns.drop(['label','pvalue'])])
    test_x['pvalue'] = test_set['pvalue']
    test_label = test_set['label']
    return train_x,train_label,test_x,test_label        

#---------------------------------------------------------------------------
def convert_chr_to_num(data):
    data['chr'].where(data['chr']!='X','23',inplace=True)
    data['chr'].where(data['chr']!='Y','24',inplace=True)
    data['chr'].where(data['chr']!='M','25',inplace=True)
    data['chr'] = data['chr'].astype('i8')
    return data

#---------------------------------------------------------------------------
def convert_chr_to_str(data):
    data['chr'].where(data['chr']!=23,'X',inplace=True)
    data['chr'].where(data['chr']!=24,'Y',inplace=True)
    data['chr'].where(data['chr']!=25,'M',inplace=True)
    data['chr'] = data['chr'].apply(lambda x: str(x))
    return data

#---------------------------------------------------------------------------

def merge_with_feature_windows(win_path,pos_sites,neg_sites):
    wincols=['chr','start','end']
    feature_wins = pd.read_csv(win_path,sep='\s+',usecols=[0,1,2],header=None,names=wincols,skiprows=1)
    feature_wins.reset_index(inplace=True)
    feature_wins['index'] = feature_wins['index']+1
    chrs = np.union1d(pos_sites['chr'].unique(),neg_sites['chr'].unique())
    chrs = [str(chrm) for chrm in chrs ]
    feature_wins['chr'] = feature_wins['chr'].apply(lambda x: x[3:])
    feature_wins = feature_wins.query('chr in @chrs')
    
    feature_wins = convert_chr_to_num(feature_wins)   
    feature_wins.sort_values(['chr','start'],inplace=True)
    pos_sites.sort_values(['chr','coordinate'],inplace=True)
    neg_sites.sort_values(['chr','coordinate'],inplace=True)
    pos_sites['start'] = (pos_sites['coordinate']/200.0).apply(lambda x: int(np.ceil(x-1))*200+1)
    neg_sites['start'] = (neg_sites['coordinate']/200.0).apply(lambda x: int(np.ceil(x-1))*200+1)
    pos_with_winid = pd.merge(pos_sites,feature_wins, on=['chr','start'],how='left')
    pos_with_winid.rename(columns={'index':'winid'},inplace=True)
    neg_with_winid = pd.merge(neg_sites,feature_wins, on=['chr','start'],how='left')
    neg_with_winid.rename(columns={'index':'winid'},inplace=True)
    return pos_with_winid,neg_with_winid


def cross_validate_score(estimator,X,y=None,sample_weight=None,cv=3):
    X = pd.DataFrame(X).copy().reset_index(drop=True)
    y = pd.Series(y).copy().reset_index(drop=True)
    sample_weight = pd.Series(sample_weight).copy().reset_index(drop=True)
    skfolds = StratifiedKFold(n_splits=cv)
    scores = []
    i = 0
    for train_index, test_index in skfolds.split(X,y):
        i += 1
        print('In {}th cross validation'.format(i))       
        clone_est = clone(estimator)
        x_train_fold = X.ix[train_index,:]
        y_train_fold = y[train_index]
        weight_train_fold = sample_weight[train_index]
        x_test_fold = X.ix[test_index]
        y_test_fold = y[test_index]
        weight_test_fold = sample_weight[test_index]
        clone_est.fit(x_train_fold,y_train_fold,weight_train_fold)
        score = clone_est.score(x_test_fold,y_test_fold,weight_test_fold)
        scores.extend([score])
    return np.mean(scores)

#------------------------------------------------------------------------------
def upSampling(X,fold):
    x = X.copy()
    temp_x = np.array(x).tolist()*fold
    temp_x = pd.DataFrame(temp_x,columns=x.columns)
    return temp_x

