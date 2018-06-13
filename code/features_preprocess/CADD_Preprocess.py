#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:27:39 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import get_winid
import pysam
from sklearn.base import BaseEstimator,TransformerMixin
#----------------------------------------------------

class CADD_Preprocess(BaseEstimator,TransformerMixin):
    
    def __init__(self,data_dir = extra_storage+'CADD/',
                 sites_file = home+'data/all_sites_winid.csv',
                 additional_feature_file = home+'data/features/addtional_features'):
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file
        
    def process(self):
        all_sites = pd.read_csv(self.sites_file)
        all_sites = get_winid.convert_chr_to_num(all_sites)
        CADD_scores = []
        CADD_file = self.data_dir+'whole_genome_SNVs.tsv.gz'
        tabix = pysam.Tabixfile(CADD_file)
        i = 0
        for site in all_sites.values:
            raw_scores_one_site = []
            phred_one_site = []
            chrm = str(site[1])
            pos = int(site[2])
            left = pos
            right = pos-1
            while len(raw_scores_one_site) == 0:
                left = left-1
                right = right+1
                #print(chrm,left,right)
                for row in tabix.fetch(chrm,left,right,parser=pysam.asTuple()):
                    raw_scores_one_site.extend([float(row[-2])])
                    phred_one_site.extend([float(row[-1])])
            average_raw = np.mean(raw_scores_one_site)
            max_raw = np.max(raw_scores_one_site)
            average_phred = np.mean(phred_one_site)
            max_phred = np.max(phred_one_site)
            CADD_scores.extend([[chrm,pos,max_raw,average_raw,max_phred,average_phred]])
            i+=1
            if i%1000 == 0:
                print([chrm,pos,max_raw,average_raw,max_phred,average_phred])
        
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['CADD'] = pd.DataFrame(CADD_scores,columns=['chr','coordinate','CADD_max_raw','CADD_avg_raw','CADD_max_phred','CADD_avg_phred'])        
        
        
                
    
#


#data_dir = '/Users/Xiaobo/Desktop/test.tsv'
#sites_file = '/Users/Xiaobo/Jobs/CpG/data/all_sites_winid.csv'
#win_path = '/home/ec2-user/CpGPython/data/wins.txt'


#all_sites.sort_values(['chr','coordinate'],inplace=True)

#wins = get_winid.read_wins(win_path,chrs)

