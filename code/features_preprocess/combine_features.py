#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:03:29 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import get_winid
#######################################################################
def nearest_tss(tss,sites_df):
    merged = pd.merge(sites_df,tss,how='outer',on=['chr','coordinate'])
    merged.sort_values(['chr','coordinate'],inplace=True)
    merged.rename(columns={'strand':'before_tss'},inplace=True)
    merged.ix[merged['before_tss'].isnull()==False, 'before_tss'] = merged.ix[merged['before_tss'].isnull()==False,'coordinate']
    merged['after_tss'] = merged['before_tss']
    merged['before_tss'].fillna(method='ffill', inplace=True)
    merged['after_tss'].fillna(method='bfill',inplace=True)
    merged['dist_to_before_tss'] = np.abs(merged['coordinate']-merged['before_tss'])
    merged['dist_to_after_tss'] = np.abs(merged['coordinate']-merged['after_tss'])
    merged['tss'] = None
    before_ix = (merged['dist_to_before_tss'] < merged['dist_to_after_tss']) | (merged['dist_to_after_tss'].isnull())
    merged.ix[before_ix,'tss'] = merged.ix[before_ix,'before_tss']
    after_ix = (merged['dist_to_before_tss'] >= merged['dist_to_after_tss']) | (merged['dist_to_before_tss'].isnull())
    merged.ix[after_ix,'tss'] = merged.ix[after_ix,'after_tss']
    merged['dist_to_nearest_tss'] = np.abs(merged['coordinate']-merged['tss']) 
    merged.dropna(axis=0,subset=['id'],inplace=True)
    return merged

#-------------------------------------------------------------------------

home = '/home/ec2-user/CpGPython/'
with pd.HDFStore(home+'all_sites_winid','r') as h5s:
    all_sites = h5s['all_sites_winid'] 
chrs = all_sites['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
all_sites.sort_values(['chr','coordinate'],inplace=True)
all_sites = nearest_tss(tss,all_sites)
