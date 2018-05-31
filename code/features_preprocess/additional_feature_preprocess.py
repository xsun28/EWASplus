#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:17:28 2018

@author: Xiaobo
"""


import os
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
from feature_preprocess import ATAC_binning
from feature_preprocess import ATAC_Preprocess, CADD_Preprocess,DANN_Preprocess,Eigen_Preprocess,GenoCanyon_Preprocess
import subprocess

subprocess.call('/home/ec2-user/CpGPython/code/feature_preprocess/Feature_export.R')

ATAC_h5s = '/home/ec2-user/CpGPython/data/ATAC_H5S'
if os.path.exists(ATAC_h5s):
    atac_process = ATAC_Preprocess.ATAC_Preprocessing()
    atac_process.process()
else:
    atac_binning = ATAC_binning.ATAC_binning()
    atac_binning.binning()
    atac_process = ATAC_Preprocess.ATAC_Preprocessing()
    atac_process.process()

cadd_preprocess = CADD_Preprocess.CADD_Preprocess()
cadd_preprocess.process()

dann_preprocess = DANN_Preprocess.DANN_Preprocess()
dann_preprocess.process()

eigen_preprocess = Eigen_Preprocess.Eigen_Preprocess()
eigen_preprocess.process()

genocanyon_scores = '/home/ec2-user/extra_storage/CpG_EWAS/GenoCanyon/Results/selected_site_scores.txt'
if os.path.exists(genocanyon_scores):
    genocanyon_preprocess = GenoCanyon_Preprocess.GenoCanyon_Preprocess()
    genocanyon_preprocess.process()
else:
    print('Run GenoCanyon R script first')
 

 