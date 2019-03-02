#running script run as: python AD_sites_selection.py

import sys
import pandas as pd
import numpy as np
from common import commons
home = commons.home
logger = commons.logger
import os
from heapq import nsmallest

def cal_beta(beta_file,pos_file):
    betas = pd.read_csv(beta_file,sep='\s+',index_col=['TargetID'])
    mean_betas = pd.DataFrame(betas.mean(axis=1),columns=['beta'])
    mean_betas.index = betas.index
    pos = pd.read_csv(pos_file,sep='\s+',usecols=[0,2,3],index_col=0, header=None,skiprows=1,names=['id','chr','coordinate'])
    beta_pos = mean_betas.join(pos)
    return beta_pos



type_name = commons.type_name  ## amyloid, cerad, tangles
with_cell_type = commons.with_cell_type ## with or without
logger.info('AD {}_{} training sites selection'.format(type_name,with_cell_type))
beta_file = home+'data/AD_CpG/ROSMAP_arrayMethylation_imputed.tsv'
pos_file = home+'data/AD_CpG/ROSMAP_arrayMethylation_metaData.tsv'
logger.info('calculating mean beta values for each AD site')
logger.info('beta values for each sample are saved in {}'.format(beta_file))
all_sites_betas = cal_beta(beta_file,pos_file)
pos_pvalues ={'amyloid':0.00005,'cerad':0.00001,'ceradaf':0.00005,'tangles':0.0000005,'cogdec':0.00003,'gpath':0.00001,'braak':0.00005}
### 0.001 for amyloid, 0.0001 for cerad, 0.00001 for tangles,0.002 for cogdec, 0.0002 for gpath,0.0002 for braak
pos_pvalue = pos_pvalues[type_name] 
logger.info('positive training sites pvalue threshold is <= {}'.format(pos_pvalue))
neg_pvalue = 0.4
logger.info('negative training sites pvalue threshold is >= {}'.format(neg_pvalue))
sample_ratio_neg_to_pos = 10
logger.info('ratio of negative training samples to positive samples are '.format(sample_ratio_neg_to_pos))



all_sites_file = home+'data/AD_CpG/Rosmap_'+type_name+'_ewas_'+with_cell_type+'celltype.csv'
logger.info('pvalues for AD {} sites are in {}'.format(type_name,all_sites_file))
all_sites = pd.read_csv(all_sites_file,usecols=[1,2,3],header=None,skiprows=1,index_col=0,names=['id','beta_sign','pvalue'])
all_sites = all_sites.join(all_sites_betas).dropna()
all_sites.reset_index(inplace=True)
all_sites = all_sites[['id','chr','coordinate','beta_sign','pvalue','beta']]
all_sites['chr'] = all_sites['chr'].astype('i8')
all_sites = all_sites.query('chr<23')
all_sites.sort_values(['pvalue'],inplace=True,ascending=True)
positive_sites = all_sites.query('pvalue<=@pos_pvalue')
positive_sites['label'] = np.where(positive_sites['beta_sign']>0,1,-1)
negative_sites = all_sites.query('pvalue>@neg_pvalue')
negative_sites['label'] = 0
negatives_sort_by_beta = negative_sites.sort_values(['beta'])

select_negs_list = []
hyper_sites = negatives_sort_by_beta.query('beta_sign>=0')
hypo_sites = negatives_sort_by_beta.query('beta_sign<0')
logger.info('sampling 10 negative (control) training sites with closest beta values to each positive training sites')
for beta,beta_sign in positive_sites[['beta','beta_sign']].values:
    tmp_sites = hyper_sites if beta_sign >=0 else hypo_sites
    negs = tmp_sites.loc[nsmallest(10, tmp_sites.index.values, key=lambda i: abs(tmp_sites.loc[i,'beta']-beta)),:]
    select_negs_list.extend(negs.values)
select_negs = pd.DataFrame(select_negs_list,columns=['id','chr','coordinate','beta_sign','pvalue','beta','label']).drop_duplicates(['chr','coordinate'])

win_path = home+'data/commons/wins.txt'
logger.info('calculating 200bp window ids of positive and negative training sites')
pos_sites_with_winid, neg_sites_with_winid = commons.merge_with_feature_windows(win_path,positive_sites,select_negs)
all_sites_with_winid = pos_sites_with_winid.append(neg_sites_with_winid,ignore_index=True)
all_sites_with_winid.drop_duplicates(['id'],inplace=True)
all_sites_with_winid.sort_values(['chr','coordinate'],inplace=True) 


#export all features to file
if not os.path.exists(home+'data/AD_CpG/'+type_name+with_cell_type):
    os.mkdir(home+'data/AD_CpG/'+type_name+with_cell_type)
with pd.HDFStore(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_sites_winid','w') as h5s:
    h5s['all_sites_winid'] = all_sites_with_winid 
    logger.info('training sites with windows id are saved as {}'.format(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_sites_winid'))
all_sites_with_winid.to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_sites_winid.csv',index=False)  
all_sites_with_winid['winid'].to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/selected_pos_winid.csv',index=False)
logger.info('training sites windows id are saved as {}'.format(home+'data/AD_CpG/'+type_name+with_cell_type+'/selected_pos_winid.csv'))


##export winid with all 450k sites
logger.info('exporting all 450k window id of AD {}'.format(type_name))
all_sites = pd.read_csv(all_sites_file,usecols=[1,2,3],header=None,skiprows=1,index_col=0,names=['id','beta_sign','pvalue'])
all_sites = all_sites.join(all_sites_betas).dropna()
all_sites.reset_index(inplace=True)
all_sites = all_sites[['id','chr','coordinate','beta_sign','pvalue','beta']]
all_sites['chr'] = all_sites['chr'].astype('i8')
all_sites = all_sites.query('chr<23')
all_sites.sort_values(['pvalue'],inplace=True,ascending=True)

all_450k_sites_with_winid, __ = commons.merge_with_feature_windows(win_path,all_sites)
all_450k_sites_with_winid.drop(['beta_sign'],axis=1,inplace=True)

with pd.HDFStore(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_450k_sites_winid','w') as h5s:
    h5s['all_450k_sites_winid'] = all_450k_sites_with_winid      
all_450k_sites_with_winid.to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_450k_sites_winid.csv',index=False)
logger.info('all 450k sites of AD {} with chromsome, coordinate, windows id and pvalues are saved in {}'.format(type_name,home+'data/AD_CpG/'+type_name+with_cell_type+'/all_450k_sites_winid'))

if not os.path.exists(home+'data/AD_CpG/selected_450k_pos_winid.csv'):
    all_450k_sites_with_winid['winid'].to_csv(home+'data/AD_CpG/selected_450k_pos_winid.csv',index=False)
    logger.info('450K AD sites window ids are saved in {}'.format(home+'data/AD_CpG/selected_450k_pos_winid.csv'))
if not os.path.exists(home+'data/AD_CpG/all_450k_sites_winid.csv'):
    all_450k_sites_with_winid.to_csv(home+'data/AD_CpG/all_450k_sites_winid.csv',index=False)
    with pd.HDFStore(home+'data/AD_CpG/all_450k_sites_winid','w') as h5s:
        h5s['all_450k_sites_winid'] = all_450k_sites_with_winid 
    logger.info('450K AD sites chromsomes, coordinates, window ids are saved in {}'.format(home+'data/AD_CpG/all_450k_sites_winid.csv'))
