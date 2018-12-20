#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:54:58 2018

@author: Xiaobo"""
import pandas as pd
import numpy as np
import re
import os
import sys
import gc
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import get_winid

############################################################
class BED_binning(object):
    
    def __init__(self,data_type='ATAC',data_dir=extra_storage+'ATAC/',output=home+'data/commons/ATAC_H5S',win_path=home+'data/commons/wins.txt',chrs=np.arange(1,22,dtype='int64'),sorted=False):
        self.data_dir = data_dir
        self.output = output
        self.win_path = win_path
        self.chrs = chrs
        self.data_type = data_type
        self.sorted = sorted
    
    
#    def read_bed(self,file):
#        bed = pd.read_csv(file,usecols=[0,1,2,5],header=None,names=['chr','pos1','pos2','strand'],sep='\s+')
#        bed['chr'] = bed['chr'].apply(lambda x: 'chr'+x.split('.')[-1] if not x.startswith('chr') else x)
#        bed['coordinate'] = np.where(bed['strand']=='+',bed['pos1']+1,bed['pos2'])
#        bed.drop(['pos1','pos2'],axis=1,inplace=True)
#        bed['count'] = 1
#        print(bed['chr'].unique())
#        #    bed_counts = bed.groupby(['chr','coordinate']).aggregate({'count':sum})
#        return bed

    def cal_counts(self,h5s,file,wins):
        print('start process '+file)
        if self.data_type == 'WGBS':
            bed = self.read_WGBS((self.data_dir+file))
        else:
            bed = pd.read_csv(file,usecols=[0,1,2,5],header=None,names=['chr','pos1','pos2','strand'],sep='\s+')
            bed['chr'] = bed['chr'].apply(lambda x: 'chr'+x.split('.')[-1] if not x.startswith('chr') else x)
            #bed = self.read_bed((self.data_dir+file))
        bed = get_winid.convert_chr_to_num(bed,self.chrs)
        
        if self.data_type == 'WGBS':
            bed = pd.merge(wins,bed,left_on=['oldChr','oldCoordinate'],right_on=['chr','coordinate'],how='left').dropna()
            bed = bed.drop(['chr_y','coordinate_y','oldChr','oldCoordinate'],axis=1).rename(columns={'chr_x':'chr','coordinate_x':'coordinate'}).sort_values(['chr','coordinate']).reset_index(drop=True)
            bed_counts = bed.groupby(['winid']).aggregate({'count':np.mean}).reset_index()
        else:
            bed = get_winid.get_window_reads(wins,bed,start_index=0).dropna()#.sort_values(['winid'])
            bed_counts = bed.groupby(['winid']).aggregate({'count':sum}).reset_index()
            print(bed)
            del bed
            gc.collect()
        bed_counts.rename(columns={'count':file[:-4]+'_'+self.data_type+'_counts'},inplace=True)
        h5s[file[:-4]] = bed_counts 
        print(file+' is done')
    
    def binning(self,single_file=None):
        if self.data_type == 'WGBS':
            wins = pd.read_csv(self.win_path,usecols=['chr','coordinate','oldChr','oldCoordinate','start','winid','end'])
        else:
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

                
    def read_WGBS(self,file):
        bed = pd.read_csv(file,usecols=[0,1,2,5,9,10],header=None,names=['chr','pos1','pos2','strand','total','percent'],sep='\s+')
        bed.dropna(inplace=True)
        bed['coordinate'] = np.where(bed['strand']=='+',bed['pos1']+1,bed['pos2'])
        bed.drop(['pos1','pos2'],axis=1,inplace=True)
        bed['count'] = np.round(bed['total']*bed['percent']/100.0)
        bed.drop(['total','percent'],axis=1,inplace=True)
        bed = bed.groupby(['chr','coordinate']).aggregate({'count':sum}).reset_index()
        #    bed_counts = bed.groupby(['chr','coordinate']).aggregate({'count':sum})
        return bed



