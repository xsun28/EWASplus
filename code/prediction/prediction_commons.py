#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

tss_start = 0
tss_end = 26573858

#---------------------------------------------------------------------------
def nearest_tss(tss,sites_df):
    merged = pd.merge(sites_df,tss,how='outer',on=['chr','coordinate'])
    merged.sort_values(['chr','coordinate'],inplace=True)
    merged.rename(columns={'strand':'before_tss'},inplace=True)
    merged.loc[merged['before_tss'].isnull()==False, 'before_tss'] = merged.loc[merged['before_tss'].isnull()==False,'coordinate']
    merged['after_tss'] = merged['before_tss']
    merged['before_tss'].fillna(method='ffill', inplace=True)
    merged['after_tss'].fillna(method='bfill',inplace=True)
    merged['dist_to_before_tss'] = np.abs(merged['coordinate']-merged['before_tss'])
    merged['dist_to_after_tss'] = np.abs(merged['coordinate']-merged['after_tss'])
    merged['tss'] = None
    before_ix = (merged['dist_to_before_tss'] < merged['dist_to_after_tss']) | (merged['dist_to_after_tss'].isnull())
    merged.loc[before_ix,'tss'] = merged.loc[before_ix,'before_tss']
    after_ix = (merged['dist_to_before_tss'] >= merged['dist_to_after_tss']) | (merged['dist_to_before_tss'].isnull())
    merged.loc[after_ix,'tss'] = merged.loc[after_ix,'after_tss']
    merged['dist_to_nearest_tss'] = np.abs(merged['coordinate']-merged['tss']) 
    return merged.dropna()

