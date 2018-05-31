#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:54:58 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import get_winid

############################################################
class BED_binning(object):
    
    def __init__(self,data_type='ATAC',data_dir=extra_storage+'ATAC/',output=home+'data/ATAC_H5S',win_path=home+'data/wins.txt',chrs=np.arange(1,23,dtype='int64')):
        self.data_dir = data_dir
        self.output = output
        self.win_path = win_path
        self.chrs = chrs
        self.data_type = data_type

    def read_bed(self,file):
        bed = pd.read_csv(file,usecols=[0,1,2,5],header=None,names=['chr','pos1','pos2','strand'],sep='\s+')
        bed['coordinate'] = np.where(bed['strand']=='+',bed['pos1'],bed['pos2'])
        bed.drop(['pos1','pos2'],axis=1,inplace=True)
        bed['count'] = 1
        #    bed_counts = bed.groupby(['chr','coordinate']).aggregate({'count':sum})
        return bed

    def cal_counts(self,h5s,file,wins):
        bed = self.read_bed((self.data_dir+file))
        bed = get_winid.convert_chr_to_num(bed,self.chrs)
        bed = get_winid.get_winid(wins,bed).sort_values(['winid'])
        bed_counts = bed.groupby(['winid']).aggregate({'count':sum}).reset_index()
        bed_counts.rename(columns={'count':file[:-4]+'_'+self.data_type+'_counts'},inplace=True)
        h5s[file[:-4]] = bed_counts 
        print(file+' is done')
    
    def binning(self,single_file=None):
        wins = get_winid.read_wins(self.win_path,self.chrs)
        if single_file is None:
            files = os.listdir(self.data_dir)
            pattern = '.*\.bed$'
            reg = re.compile(pattern)
            files = [f for f in files if len(reg.findall(f))>0]
            
            with pd.HDFStore(self.output,'w') as h5s:
                for file in files:
                    self.cal_counts(h5s,file,wins)
        else:
             with pd.HDFStore(self.output+single_file,'w') as h5s:
                self.cal_counts(h5s,single_file+'.bed',wins)
        




