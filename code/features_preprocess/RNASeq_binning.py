#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:50:08 2018

@author: Xiaobo
"""


import multiprocessing as mp
import subprocess
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from feature_preprocess import BED_binning
import os

def RNASeq_Preprocessing(file):
    data_dir = extra_storage+'RNASeq/'
    script_path = data_dir+'RNASeq_download_to_bed.sh'
    RNASeq_h5s = home+'data/RNASeq/'
    single_file = file.split('/')[-1].split('.')[0]
    if not os.path.exists(data_dir+single_file+'.bed'):
        subprocess.check_call([script_path,file])
    else:
        print(single_file+'.bed already exists')
    rnaseq_binning = BED_binning.BED_binning(data_type='RNASeq',data_dir=extra_storage+'RNASeq/',output=RNASeq_h5s)
    rnaseq_binning.binning(single_file=single_file)
    os.remove(data_dir+single_file+'.bam')
    os.remove(data_dir+single_file+'.bed')
    
data_dir = extra_storage+'RNASeq/'
pool = mp.Pool(processes=3)
files = [line.rstrip('\n') for line in open(data_dir+'files.txt','r')][1:]
pool.map(RNASeq_Preprocessing,files[3:])