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
from pybedtools import BedTool
import gc
import csv
#----------------------------------------------------

#
class Eigen_Preprocess(object):
    def __init__(self,data_dir = extra_storage+'Eigen/',sites_file = home+'data/commons/all_sites_winid.csv',additional_feature_file = home+'data/features/addtional_features'):
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file 
    
    def convert2bed(chr):
        eigen_file = self.data_dir+'Eigen_hg19_noncoding_annot_chr'+chr+'.tab.bgz'
        chr_file = pd.read_csv(eigen_file, usecols=[0,1,30,32],delimiter='\t',compression='gzip',skiprows=1,header=None,names=['chr','coordinate','eigen_phred','eigen_pc_phred'])
        chr_file['end'] = chr_file['coordinate']+1
        chr_file = chr_file[['chr','coordinate','end','eigen_phred','eigen_pc_phred']]
        chr_file.to_csv(self.data_dir+'chr'+chr+'.tsv',header=None,index=False,sep='\t')
    
    def mean_max(x):
        x['eigen_max_phred'] = x['eigen_phred'].max()
        x['egien_avg_phred'] = x['eigen_phred'].mean()
        x['eigen_max_pc_phred'] = x['eigen_pc_phred'].max()
        x['egien_avg_pc_phred'] = x['eigen_pc_phred'].mean()
        return x[:1][['chr','coordinate','distiance_to_nearest_eigen','eigen_max_phred','egien_avg_phred','eigen_max_pc_phred','egien_avg_pc_phred']]
    
    def process(self):
        all_sites = pd.read_csv(self.sites_file,usecols=['chr','coordinate'])
        all_sites = get_winid.convert_chr_to_num(all_sites)
        chrs = np.sort(all_sites['chr'].unique())
        all_sites_closest = []
        for chr in chrs:
            print('processing sites on chr '+str(chr))
            chr_file = self.data_dir+'chr'+chr+'.tsv'
            if not os.path.exists(chr_file):
                self.split_by_chr()
            chr_sites = all_sites.query('chr==@chr')
            chr_sites['end'] = chr_sites['coordinate']+1
            chr_sites = BedTool([tuple(x[1]) for x in chr_sites.iterrows()])
            chr_sites_closest = chr_sites.closest(chr_file,d=True,nonamecheck=True)
            for row in chr_sites_closest:
                all_sites_closest.extend([[row[0],row[1],row[6],row[7],row[8]]])
            del chr_sites_closest
            del chr_sites
            gc.collect()
        all_sites_closest = pd.DataFrame(all_sites_closest,columns=['chr','coordinate','eigen_phred','eigen_pc_phred','distiance_to_nearest_eigen'])
        all_sites_closest = all_sites_closest.groupby(['chr','coordinate']).apply(mean_max).reset_index()
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['Eigen'] = all_sites_closest





        