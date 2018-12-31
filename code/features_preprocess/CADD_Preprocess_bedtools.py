
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:27:39 2018
@author: Xiaobo
"""

import pandas as pd
import numpy as np
import sys
import os
home = os.getcwd()[:os.getcwd().find('EnsembleCpG')]+'EnsembleCpG/'
os.environ["PYTHONPATH"] = home+"code/"
sys.path[1] = os.environ["PYTHONPATH"]
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import get_winid
import pysam
from pybedtools import BedTool
import csv
from sklearn.base import BaseEstimator,TransformerMixin
import gc
import os
#----------------------------------------------------

class CADD_Preprocess(BaseEstimator,TransformerMixin):
    
    def __init__(self,data_dir = extra_storage+'CADD/',
                 sites_file = home+'data/commons/all_sites_winid.csv',
                 additional_feature_file = home+'data/features/addtional_features'):
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file
        
    def split_by_chr():
        CADD_file = self.data_dir+'whole_genome_SNVs.tsv.gz'
        tabix = pysam.Tabixfile(CADD_file)
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
        x['CADD_max_phred'] = x['phred'].max()
        x['CADD_avg_phred'] = x['phred'].mean()
        return x[:1][['chr','coordinate','distiance_to_nearest_CADD','CADD_max_phred','CADD_avg_phred']]
    
    
    def process(self):
        all_sites = pd.read_csv(self.sites_file,usecols=['chr','coordinate'])
        all_sites = get_winid.convert_chr_to_num(all_sites)
        chrs = np.sort(all_sites['chr'].unique())
        all_sites_closest = []
        for chr in chrs:
            print('processing sites on chr '+str(chr))
            chr_file = self.data_dir+'chr'+str(chr)+'.tsv'
            if not os.path.exists(chr_file):
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
        all_sites_closest = pd.DataFrame(all_sites_closest,columns=['chr','coordinate','phred','distiance_to_nearest_CADD'])
        all_sites_closest = all_sites_closest.groupby(['chr','coordinate']).apply(self.mean_max).reset_index()
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['CADD'] = all_sites_closest


       
        
        

