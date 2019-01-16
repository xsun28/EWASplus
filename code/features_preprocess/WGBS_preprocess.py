#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:03:52 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
import os
from common import commons
home = commons.home
logger = commons.logger
extra_storage = commons.extra_storage
import os
import re
from features_preprocess import get_winid
class WGBS_Preprocess():
    def __init__(self,h5s_file= home+'data/commons/WGBS_single_H5S',data_dir = extra_storage+'WGBS/',sites_file = home+'data/commons/all_sites_winid.csv',additional_feature_file = home+'data/features/addtional_features',hg19_file= home+'data/WGBS/hg19_WGBS.csv',chrs=np.arange(1,23,dtype='int64')):
        self.h5s_file = h5s_file
        self.data_dir = data_dir
        self.sites_file = sites_file
        self.additional_feature_file = additional_feature_file         
        self.hg19_sites = pd.read_csv(hg19_file,usecols=[0,1,3,4]).query('chr in @chrs')
        self.chrs = chrs
        
    def process(self):
        files = os.listdir(self.data_dir)
        pattern = '.*\.bed$'
        reg = re.compile(pattern)
        files = [f for f in files if len(reg.findall(f))>0]
        logger.info('WGBS files with converted coordinate are saved at: '+self.h5s_file)    
        with pd.HDFStore(self.h5s_file,'w') as h5s:
            for file in files:
                logger.info("start converting coordinate of WGBS: "+file)
                self.readcount_WGBS(h5s,file)        
        
             
    def readcount_WGBS(self,h5s,file):   ####convert each hg38 WGBS file to hg19 coordinate

        bed = pd.read_csv(self.data_dir+file,usecols=[0,1,2,5,9,10],header=None,names=['chr','pos1','pos2','strand','total','percent'],sep='\s+')       
        bed.dropna(inplace=True)
        bed['coordinate'] = np.where(bed['strand']=='+',bed['pos1'],bed['pos1']-1) ##read 0-based WGBS bed, merge +/- strand
        bed.drop(['pos1','pos2'],axis=1,inplace=True)
        bed['count'] = np.round(bed['total']*bed['percent']/100.0)
        bed.drop(['total','percent'],axis=1,inplace=True)
        bed = get_winid.convert_chr_to_num(bed,self.chrs)
        bed = bed.groupby(['chr','coordinate']).aggregate({'count':sum}).reset_index()
        bed = pd.merge(self.hg19_sites,bed,left_on=['hg38chr','hg38coordinate'],right_on=['chr','coordinate'],how='left').dropna()
        bed = bed.drop(['chr_y','coordinate_y','hg38chr','hg38coordinate'],axis=1).rename(columns={'chr_x':'chr','coordinate_x':'coordinate'}).sort_values(['chr','coordinate']).reset_index(drop=True)
        bed = bed.groupby(['chr','coordinate']).aggregate({'count':sum}).reset_index()
        bed.rename(columns={'count':file[:-4]+'_WGBS_counts'},inplace=True)
        h5s[file[:-4]] = bed
        logger.info("WGBS: "+file+' is coordinate converted')
        #    bed_counts = bed.groupby(['chr','coordinate']).aggregate({'count':sum})


    def scores(self):
        all_sites = pd.read_csv(self.sites_file)
        counts_at_targets = pd.DataFrame(all_sites[['chr','coordinate']]) #.sort_values(['winid'])
        with pd.HDFStore(self.h5s_file,'r') as h5s:
            for key in h5s.keys():
                bed_counts = h5s[key].dropna()
                counts_at_targets = pd.merge(counts_at_targets,bed_counts,on=['chr','coordinate'],how='left')
                counts_at_targets[key[1:]+'_WGBS_counts'].fillna(0,inplace=True)
                logger.info('merging WGBS '+key+' file with selected sites is done')
        
        with pd.HDFStore(self.additional_feature_file,'a') as h5s:
            h5s['WGBS'] = counts_at_targets               