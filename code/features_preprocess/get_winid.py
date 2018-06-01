#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:24:47 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
###################################
def convert_chr_to_num(data,chrs=None):
    def f(x):
        x[i] = int(x[i][3:])
        return x
    if data['chr'].dtype != np.dtype('int64'):
        i = data.columns.get_loc('chr')
        if chrs is not None:
            chr_str = [str(chrm) for chrm in chrs]
            print(chr_str)
        print(data.ix[1:10,0])
        l = [f(x) for x in data.values if x[i].startswith('chr') and x[i][3:] in chr_str ]
        return pd.DataFrame(l,columns=data.columns)
    else:
        return data
#-----------------------------------
def read_wins(win_path,chrs=None):
    wincols=['chr','start','end']
    feature_wins = pd.read_csv(win_path,sep='\s+',usecols=[0,1,2],header=None,names=wincols,skiprows=1)
    feature_wins.reset_index(inplace=True)
    feature_wins['index'] = feature_wins['index']+1    
    feature_wins = convert_chr_to_num(feature_wins,chrs)   
    feature_wins.sort_values(['chr','start'],inplace=True)
    return feature_wins

#----------------------------------
def get_winid(feature_wins,dataset):

#    chrs = dataset['chr'].unique()
    dataset.sort_values(['chr','coordinate'],inplace=True)
    dataset['start'] = (dataset['coordinate']/200.0).apply(lambda x: int(np.ceil(x-1))*200+1)
    dataset_with_winid = pd.merge(dataset,feature_wins, on=['chr','start'],how='left')
    dataset_with_winid.rename(columns={'index':'winid'},inplace=True)
    return dataset_with_winid