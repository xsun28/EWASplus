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
    if data['chr'].dtype != np.dtype('int64'): 
        data['chr'].where(data['chr'].apply(lambda x: x[:3].lower()) != 'chr',data['chr'].apply(lambda x: x[3:]),inplace=True)
        data['chr'].where(data['chr']!='X','23',inplace=True)
        data['chr'].where(data['chr']!='Y','24',inplace=True)
        data['chr'].where(data['chr']!='M','25',inplace=True)
        if chrs is not None:
            chr_str = [str(chrm) for chrm in chrs]
            print(chr_str)
            data = data.query('chr in @chr_str')
        data['chr'] = data['chr'].astype('i8')
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