#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:24:47 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
from common.commons import logger
###################################
def convert_chr_to_num(data,chrs=None):
    def f(x):
        x[i] = int(x[i][3:])
        return x
    if data['chr'].dtype != np.dtype('int64'):
        if chrs is not None:
            chr_str = [str(chrm) for chrm in chrs]
            #print(chr_str)
        data = data[data['chr'].apply(lambda x: x.startswith('chr') and x[3:] in chr_str)]
        data['chr'] = data['chr'].apply(lambda x: int(x[3:]))
        #i = data.columns.get_loc('chr')
        #l = [f(x) for x in data.values if x[i].startswith('chr') and x[i][3:] in chr_str ]
        #return pd.DataFrame(l,columns=data.columns)
    return data.reset_index(drop=True)
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
def get_winid(feature_wins,dataset,sorted=False,start_index=1):
    logger.info('getting windows id for sites in the dataset with start index {}'.format(start_index))
#    chrs = dataset['chr'].unique()
    if not sorted:
        dataset.sort_values(['chr','coordinate'],inplace=True)
    dataset['start'] = np.floor((dataset['coordinate']-start_index)/200.0)*200+1
    dataset_with_winid = pd.merge(dataset,feature_wins, on=['chr','start'],how='left')
    dataset_with_winid.rename(columns={'index':'winid'},inplace=True)
    return dataset_with_winid

#-------------------------------------
def get_window_reads(feature_wins,dataset,start_index=1):
    dataset['window_start'] = np.floor((dataset['pos1']-start_index)/200.0)*200+1
    dataset['window_end'] = dataset['window_start']+200
    dataset['window_start_reads'] = np.where(dataset['window_end']>=dataset['pos2'],1,(dataset['window_end']-dataset['pos1'])/(dataset['pos2']-dataset['pos1']))
    dataset['window_end_reads'] = np.where(dataset['window_end']>=dataset['pos2'],0,(dataset['pos2']-dataset['window_end'])/(dataset['pos2']-dataset['pos1']))
    dataset = pd.DataFrame(dataset[['chr','window_start','window_start_reads']].values.tolist()+dataset[['chr','window_end','window_end_reads']].values.tolist(),columns=['chr','start','count']).sort_values(['chr','start'])
    dataset = pd.merge(dataset,feature_wins, on=['chr','start'],how='left')
    dataset.rename(columns={'index':'winid'},inplace=True)
    return dataset
