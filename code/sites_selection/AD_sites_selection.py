#running script run as: python AD_sites_selection.py

import sys
import pandas as pd
import numpy as np
from common import commons
home = commons.home
from log import Logger
import os

def cal_beta(beta_file,pos_file):
    betas = pd.read_csv(beta_file,sep='\s+',index_col=['TargetID'])
    mean_betas = pd.DataFrame(betas.mean(axis=1),columns=['beta'])
    mean_betas.index = betas.index
    pos = pd.read_csv(pos_file,sep='\s+',usecols=[0,2,3],index_col=0, header=None,skiprows=1,names=['id','chr','coordinate'])
    beta_pos = mean_betas.join(pos)
    return beta_pos



type_name = commons.type_name  ## amyloid, cerad, tangles
with_cell_type = commons.with_cell_type ## with or without
log_dir = home+'logs/'
logger = Logger.Logger(log_dir).get_logger()
beta_file = home+'data/AD_CpG/ROSMAP_arrayMethylation_imputed.tsv'
pos_file = home+'data/AD_CpG/ROSMAP_arrayMethylation_metaData.tsv'
all_sites_betas = cal_beta(beta_file,pos_file)
pos_pvalues ={'amyloid':0.00005,'cerad':0.00001,'ceradaf':0.00005,'tangles':0.0000005,'cogdec':0.00003,'gpath':0.00001,'braak':0.00005}
### 0.001 for amyloid, 0.0001 for cerad, 0.00001 for tangles,0.002 for cogdec, 0.0002 for gpath,0.0002 for braak
pos_pvalue = pos_pvalues[type_name] 
neg_pvalue = 0.4
sample_ratio_neg_to_pos = 10

all_sites_file = home+'data/AD_CpG/Rosmap_'+type_name+'_ewas_'+with_cell_type+'celltype.csv'
all_sites = pd.read_csv(all_sites_file,usecols=[1,2,3],header=None,skiprows=1,index_col=0,names=['id','beta_sign','pvalue'])
all_sites = all_sites.join(all_sites_betas).dropna()
all_sites.reset_index(inplace=True)
temp = pd.DataFrame()
temp['id'],temp['chr'],temp['coordinate'],temp['beta_sign'],temp['pvalue'],temp['beta'] = all_sites['id'],all_sites['chr'],all_sites['coordinate'],all_sites['beta_sign'],all_sites['pvalue'],all_sites['beta']
all_sites = temp
all_sites['chr'] = all_sites['chr'].astype('i8')
all_sites.sort_values(['pvalue'],inplace=True,ascending=True)
positive_sites = all_sites.query('pvalue<=@pos_pvalue')
positive_sites['label'] = np.where(positive_sites['beta_sign']>0,1,-1)
negative_sites = all_sites.query('pvalue>@neg_pvalue')
negative_sites['label'] = 0
negatives_sort_by_beta = negative_sites.sort_values(['beta'])

select_negs_list = []
hyper_sites = negatives_sort_by_beta.query('beta_sign>=0')
hypo_sites = negatives_sort_by_beta.query('beta_sign<0')
for beta,beta_sign in positive_sites[['beta','beta_sign']].values:
    tmp_sites = hyper_sites if beta_sign >=0 else hypo_sites
    neg_ix = tmp_sites['beta'].searchsorted(beta)[0]    
    negs = tmp_sites.iloc[neg_ix-int(sample_ratio_neg_to_pos/2):np.minimum(neg_ix+int(sample_ratio_neg_to_pos/2),len(negatives_sort_by_beta)),:]
    select_negs_list.extend(negs.values)
select_negs = pd.DataFrame(select_negs_list,columns=['id','chr','coordinate','beta_sign','pvalue','beta','label'])

win_path = home+'data/commons/wins.txt'
pos_sites_with_winid, neg_sites_with_winid = commons.merge_with_feature_windows(win_path,positive_sites,select_negs)
all_sites_with_winid = pos_sites_with_winid.append(neg_sites_with_winid,ignore_index=True)
all_sites_with_winid.drop_duplicates(['id'],inplace=True)
all_sites_with_winid.sort_values(['chr','coordinate'],inplace=True) 


#export all features to file
if not os.path.exists(home+'data/AD_CpG/'+type_name+with_cell_type):
    os.mkdir(home+'data/AD_CpG/'+type_name+with_cell_type)
with pd.HDFStore(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_sites_winid','w') as h5s:
    h5s['all_sites_winid'] = all_sites_with_winid       
all_sites_with_winid.to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_sites_winid.csv',index=False)  
all_sites_with_winid['winid'].to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/selected_pos_winid.csv',index=False)



##export winid with all 450k sites
all_sites_file = home+'data/AD_CpG/Rosmap_'+type_name+'_ewas_'+with_cell_type+'celltype.csv'
all_sites = pd.read_csv(all_sites_file,usecols=[1,2,3],header=None,skiprows=1,index_col=0,names=['id','beta_sign','pvalue'])
all_sites = all_sites.join(all_sites_betas).dropna()
all_sites.reset_index(inplace=True)
temp = pd.DataFrame()
temp['id'],temp['chr'],temp['coordinate'],temp['beta_sign'],temp['pvalue'],temp['beta'] = all_sites['id'],all_sites['chr'],all_sites['coordinate'],all_sites['beta_sign'],all_sites['pvalue'],all_sites['beta']
all_sites = temp
all_sites['chr'] = all_sites['chr'].astype('i8')
all_sites.sort_values(['pvalue'],inplace=True,ascending=True)

all_450k_sites_with_winid, __ = commons.merge_with_feature_windows(win_path,all_sites)
all_450k_sites_with_winid.drop(['beta_sign'],axis=1,inplace=True)

with pd.HDFStore(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_450k_sites_winid','w') as h5s:
    h5s['all_450k_sites_winid'] = all_450k_sites_with_winid      
all_450k_sites_with_winid.to_csv(home+'data/AD_CpG/'+type_name+with_cell_type+'/all_450k_sites_winid.csv',index=False)  
if not os.path.exists(home+'data/AD_CpG/selected_450k_pos_winid.csv'):
    all_450k_sites_with_winid['winid'].to_csv(home+'data/AD_CpG/selected_450k_pos_winid.csv',index=False)
if not os.path.exists(home+'data/AD_CpG/all_450k_sites_winid.csv'):
    all_450k_sites_with_winid.to_csv(home+'data/AD_CpG/all_450k_sites_winid.csv',index=False)
    with pd.HDFStore(home+'data/AD_CpG/all_450k_sites_winid','w') as h5s:
        h5s['all_450k_sites_winid'] = all_450k_sites_with_winid 