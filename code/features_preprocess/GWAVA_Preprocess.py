#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:23:38 2018

@author: Xiaobo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:42:06 2018

@author: Xiaobo
"""

import pandas as pd
import numpy as np
import argparse
import re
import os
import sys
sys.path.append('/home/ec2-user/CpGPython/code/')
import get_winid
###########################

#----------------------------


##############################   
#parser = argparse.ArgumentParser(description='ATAC processor')
#parser.add_argument('-i',required=True,help='input file directory path',dest='input',metavar='input dir',default='/home/ec2-user/extra_storage/CpG_EWAS/GWAVA/csv/')
#parser.add_argument('-s',required=True,help='all sites with winid path',dest='sites',metavar='all sites winid',default='/home/ec2-user/all_sites_winid.csv')
#parser.add_argument('-w',required=True,help='window file',dest='win',metarvar='window file',default='/home/ec2-user/wins.txt')
#args = parser.parse_args()
#data_dir = args.input
#sites_file = args.sites
#win_path = args.win

data_dir = '/home/ec2-user/extra_storage/CpG_EWAS/GWAVA/csv/'
sites_file = '/home/ec2-user/CpGPython/data/all_sites_winid.csv'
win_path = '/home/ec2-user/CpGPython/data/wins.txt'
additional_feature_file = '/home/ec2-user/CpGPython/data/features/addtional_features'
#----------------------------------------
all_sites = pd.read_csv(sites_file)
all_sites = get_winid.convert_chr_to_num(all_sites)
all_sites.sort_values(['winid'],inplace=True)
chrs = all_sites['chr'].unique()
wins = get_winid.read_wins(win_path,chrs)
#-----------------------------------------
files = os.listdir(data_dir)
pattern = '^scores.*\.csv$'
reg = re.compile(pattern)
files = [f for f in files if len(reg.findall(f))>0]
gwava_at_targets_list = []
for f in files:
    gwava = pd.read_csv(data_dir+f,usecols=[1,3,179,180,181],skiprows=1,header=None,names=['chr','coordinate','gwava_region_score','gwava_tss_score','gwava_unmatched_score'])
    gwava = get_winid.convert_chr_to_num(gwava,chrs)
    gwava = get_winid.get_winid(wins,gwava).sort_values(['winid'])
    gwava.rename(columns={'coordinate':'gwava_pos'},inplace=True)
    gwava_targets = pd.merge(all_sites.drop(['pvalue','beta','label'],axis=1),gwava.drop(['chr','start','end'],axis=1),on=['winid'],how='left').dropna(axis=0)
    if len(gwava_targets) > 0:
        gwava_targets = gwava_targets.groupby(['id']).apply(lambda x: x.ix[np.argmin(np.abs(x['coordinate']-x['gwava_pos']))]).drop(['id'],axis=1).reset_index()
        gwava_at_targets_list.extend(gwava_targets.dropna(axis=0).values)

gwava_at_targets = pd.DataFrame(gwava_at_targets_list,columns=['id','chr','coordinate','start','winid','end','gwava_pos','gwava_region_score','gwava_tss_score','gwava_unmatched_score'])    
with pd.HDFStore(additional_feature_file,'w') as h5s:
    h5s['GWAVA'] = gwava_at_targets