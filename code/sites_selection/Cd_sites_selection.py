#running script: run as python Cd_sites_selection.py
import pandas as pd
import numpy as np
import sys
from common import commons
home_path = commons.home
from log import Logger

log_dir = home_path+'logs/'
logger = Logger.Logger(log_dir,True).get_logger()
pos_pvalue = 0.0003
neg_pvalue = 0.1
sample_ratio_neg_to_pos = 10


all_sites = pd.read_excel(home_path+'data/Cd/allsites.xlsx','Excel Table S4',skiprows=4,header=None, names=['id','chr','coordinate','beta_sign','pvalue'],usecols=[0,1,2,5,6])
logger.info('Datasets location: '+home_path+'data/Cd/allsites.xlsx')
betas = pd.read_csv(home_path+'data/Cd/RICHS_betaValue_summary.csv',skiprows=1,header=None,usecols=[0,1],names=['id','beta_mean'])
all_sites.sort_values(['id'],inplace=True)
betas.sort_values(['id'],inplace=True)
all_sites = pd.merge(all_sites,betas,on=['id'],how='left')
all_sites.rename(columns={'beta_mean':'beta'},inplace=True)
all_sites.sort_values(['pvalue'],inplace=True,ascending=True)
positive_sites = all_sites.query('pvalue<=@pos_pvalue')
positive_sites['label'] = np.where(positive_sites['beta_sign']>0,1,-1)
negative_sites = all_sites.query('pvalue>@neg_pvalue')
negative_sites['label'] = 0


select_negs_list = []
negatives_sort_by_beta = negative_sites.sort_values(['beta'])
hyper_sites = negatives_sort_by_beta.query('beta_sign>=0')
hypo_sites = negatives_sort_by_beta.query('beta_sign<0')
for beta,beta_sign in positive_sites[['beta','beta_sign']].values:
    tmp_sites = hyper_sites if beta_sign >=0 else hypo_sites
    neg_ix = tmp_sites['beta'].searchsorted(beta)[0]    
    negs = tmp_sites.iloc[neg_ix-int(sample_ratio_neg_to_pos/2):np.minimum(neg_ix+int(sample_ratio_neg_to_pos/2),len(negatives_sort_by_beta)),:]
    select_negs_list.extend(negs.values)
select_negs = pd.DataFrame(select_negs_list,columns=['id','chr','coordinate','beta_sign','pvalue','beta','label'])

win_path = home_path+'data/commons/wins.txt'
pos_sites_with_winid, neg_sites_with_winid = commons.merge_with_feature_windows(win_path,positive_sites,select_negs)
all_sites_with_winid = pos_sites_with_winid.append(neg_sites_with_winid,ignore_index=True)
all_sites_with_winid.drop_duplicates(['id'],inplace=True)
all_sites_with_winid.sort_values(['chr','coordinate'],inplace=True) 

with pd.HDFStore(home_path+'data/Cd/all_sites_winid','w') as h5s:
    h5s['all_sites_winid'] = all_sites_with_winid
       
all_sites_with_winid.to_csv(home_path+'data/Cd/all_sites_winid.csv',index=False)  
all_sites_with_winid['winid'].to_csv(home_path+'data/Cd/selected_pos_winid.csv',index=False)

##export winid with all 450k sites
all_450k_sites_with_winid, __ = commons.merge_with_feature_windows(win_path,all_sites)
all_450k_sites_with_winid.drop(['beta_sign'],axis=1,inplace=True)

all_450k_sites_with_winid.to_csv(home_path+'data/Cd/all_450k_sites_winid.csv',index=False) 
all_450k_sites_with_winid['winid'].to_csv(home_path+'data/Cd/selected_450k_pos_winid.csv',index=False)
with pd.HDFStore(home_path+'data/Cd/all_450k_sites_winid','w') as h5s:
    h5s['all_450k_sites_winid'] = all_450k_sites_with_winid