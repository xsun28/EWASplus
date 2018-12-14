#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:12:33 2018
@author: Xiaobo
"""

import pandas as pd
import numpy as np
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import get_winid
import pysam
import gc
from pybedtools import BedTool
import os
import csv
################################################################

class DANN_Preprocess(object):
    
    def __init__(self,data_dir = extra_storage+'DANN/',
                 sites_file = home+'data/commons/all_sites_winid.csv',
                 additional_feature_file = home+'data/features/addtional_features'):
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file
    
    def split_by_chr():
        dann_file = self.data_dir+'DANN_whole_genome_SNVs.tsv.bgz'
        tabix = pysam.Tabixfile(dann_file)
        chrs = [str(x) for x in np.arange(1,22)]
        left = 0
        right = 2000000000
        for chr in chrs:
            chr_file = self.data_dir+'chr'+chr+'.tsv'
            with open(chr_file,'w') as f:
                writer = csv.writer(f,delimiter='\t')
                for row in tabix.fetch(chr,left,right):
                    writer.writerow((row[0],int(row[1]),int(row[1])+1,float(row[-1])))
    
    def mean_max(x):
        x['DANN_max_score'] = x['score'].max()
        x['DANN_avg_score'] = x['score'].mean()   
        return x[:1][['chr','coordinate','distiance_to_nearest_DANN','DANN_max_score','DANN_avg_score']]
    
    def process(self):
        all_sites = pd.read_csv(self.sites_file,usecols=['chr','coordinate'])
        all_sites = get_winid.convert_chr_to_num(all_sites)
        chrs = np.sort(all_sites['chr'].unique())
        all_sites_closest = []
        for chr in chrs:
            print('processing sites on chr '+str(chr))
            chr_file = self.data_dir+'chr'+str(chr)+'.tsv'
            if not os.path.exists(self.data_dir+'chr1.tsv'):
                self.split_by_chr()
            chr_sites = all_sites.query('chr==@chr')
            chr_sites['coordinate'] = chr_sites['coordinate'].astype('i8')
            chr_sites['end'] = chr_sites['coordinate']+1
            chr_sites = BedTool([tuple(x[1]) for x in chr_sites.iterrows()])
            chr_sites_closest = chr_sites.closest(chr_file,d=True,nonamecheck=True)
            for row in chr_sites_closest:
                all_sites_closest.extend([[row[0],row[1],row[6],row[7]]])
            del chr_sites_closest
            del chr_sites
            gc.collect()
        all_sites_closest = pd.DataFrame(all_sites_closest,columns=['chr','coordinate','score','distiance_to_nearest_DANN'])
        all_sites_closest = all_sites_closest.groupby(['chr','coordinate']).apply(mean_max).reset_index()
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['DANN'] = all_sites_closest    
        
        
