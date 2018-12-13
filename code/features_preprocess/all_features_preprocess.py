import os
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
from features_preprocess import BED_binning
from features_preprocess import BED_Preprocess, CADD_Preprocess,DANN_Preprocess,Eigen_Preprocess,GenoCanyon_Preprocess,WGBS_preprocess
import subprocess
from features_preprocess import get_winid
from common.commons import rename_features
import os
import re
import gc

def nearest_tss(tss,sites_df):
    merged = pd.merge(sites_df,tss,how='outer',on=['chr','coordinate'])
    merged.sort_values(['chr','coordinate'],inplace=True)
    merged.rename(columns={'strand':'before_tss'},inplace=True)
    merged.ix[merged['before_tss'].isnull()==False, 'before_tss'] = merged.ix[merged['before_tss'].isnull()==False,'coordinate']
    merged['after_tss'] = merged['before_tss']
    merged['before_tss'].fillna(method='ffill', inplace=True)
    merged['after_tss'].fillna(method='bfill',inplace=True)
    merged['dist_to_before_tss'] = np.abs(merged['coordinate']-merged['before_tss'])
    merged['dist_to_after_tss'] = np.abs(merged['coordinate']-merged['after_tss'])
    merged['tss'] = None
    before_ix = (merged['dist_to_before_tss'] < merged['dist_to_after_tss']) | (merged['dist_to_after_tss'].isnull())
    merged.ix[before_ix,'tss'] = merged.ix[before_ix,'before_tss']
    after_ix = (merged['dist_to_before_tss'] >= merged['dist_to_after_tss']) | (merged['dist_to_before_tss'].isnull())
    merged.ix[after_ix,'tss'] = merged.ix[after_ix,'after_tss']
    merged['dist_to_nearest_tss'] = np.abs(merged['coordinate']-merged['tss']) 
    merged.drop(['before_tss','after_tss','tss','dist_to_before_tss','dist_to_after_tss'],axis=1,inplace=True)
    merged.dropna(axis=0,subset=['id'],inplace=True)
    return merged



parser = argparse.ArgumentParser(description='Adding all features to all selected sites')
parser.add_argument('-d',required=True,default='AD_CpG',help='disease dataset',dest='dataset',metavar='AD or Cd?')
args = parser.parse_args()
dataset = args.dataset # AD_CpG or Cd
if dataset == 'AD_CpG':
    type_name = commons.type_name  ## amyloid, cerad, tangles
    with_cell_type = commons.with_cell_type ## with or without
    dataset = dataset+'/'+type_name+with_cell_type
sites_file = home+'data/'+dataset+'/all_sites_winid.csv'
additional_feature_file = home+'data/features/'+dataset+'/addtional_features'

subprocess.call([home+'code/features_preprocess/Feature_export.R',home+'data',dataset,'False'])

#single sites WGBS
all_wgbs_sites_file = home+'data/WGBS/all_wgbs_sites_winid.csv'
WGBS_h5s = home+'data/commons/WGBS_single_H5S'
WGBS_proc = WGBS_preprocess.WGBS_Preprocess(h5s_file=WGBS_h5s,data_dir=extra_storage+'WGBS/',sites_file=sites_file,additional_feature_file=additional_feature_file,hg19_file= home+'data/WGBS/hg19_WGBS.csv')
if not os.path.exists(WGBS_h5s):
    WGBS_proc.process()
WGBS_proc.scores()

ATAC_h5s = home+'data/commons/ATAC_H5S'
if os.path.exists(ATAC_h5s):
    atac_process = BED_Preprocess.BED_Preprocessing(h5s_file=ATAC_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file,data_type='ATAC')
    atac_process.process()
else:
    atac_binning = BED_binning.BED_binning(data_type='ATAC',data_dir=extra_storage+'ATAC/',output=ATAC_h5s,sorted=True)
    atac_binning.binning()
    atac_process = BED_Preprocess.BED_Preprocessing(h5s_file=ATAC_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file,data_type='ATAC')
    atac_process.process() 
    
subprocess.call(['python',home+'code/features_preprocess/RNASeq_binning.py'])


RNASeq_h5s = home+'data/RNASeq/'
if len(os.listdir(RNASeq_h5s))>0:
    rnaseq_process = BED_Preprocess.BED_Preprocessing(h5s_file=RNASeq_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file, data_type='RNASeq')
    rnaseq_process.process()
else:
    subprocess.call(['python',home+'code/features_preprocess/RNASeq_binning.py'])
    rnaseq_process = BED_Preprocess.BED_Preprocessing(h5s_file=RNASeq_h5s,sites_file=sites_file,additional_feature_file=additional_feature_file, data_type='RNASeq')
    rnaseq_process.process()
    
cadd_preprocess = CADD_Preprocess.CADD_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
cadd_preprocess.process()

dann_preprocess = DANN_Preprocess.DANN_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
dann_preprocess.process()

eigen_preprocess = Eigen_Preprocess.Eigen_Preprocess(sites_file=sites_file,additional_feature_file=additional_feature_file)
eigen_preprocess.process()

genocanyon_scores = extra_storage+'GenoCanyon/Results/'+dataset+'/selected_site_scores.txt'
data_dir=extra_storage+'GenoCanyon/Results/'+dataset+'/'
if os.path.exists(genocanyon_scores):
    genocanyon_preprocess = GenoCanyon_Preprocess.GenoCanyon_Preprocess(data_dir=data_dir,sites_file=sites_file,additional_feature_file=additional_feature_file)
    genocanyon_preprocess.process('selected_site_scores.txt')
else:
    print('Running GenoCanyon R script...')
    subprocess.call([home+'code/features_preprocess/GenoCanyon_Preprocess.R',"FALSE",home,extra_storage,dataset])
    
with pd.HDFStore(home+'data/'+dataset+'/all_sites_winid','r') as h5s:
    all_sites = h5s['all_sites_winid']
all_sites.reset_index(drop=True,inplace=True)    

gc.collect()

feature_dir = home+'data/features/'+dataset+'/'
files = os.listdir(feature_dir)
pattern = '.*all.csv$'
reg = re.compile(pattern)
files = [name for name in files if len(reg.findall(name))>0]

for file in files:    
    feature = pd.read_csv(feature_dir+file)
    print(len(feature.columns))
    all_sites = pd.concat([all_sites,feature],axis=1)

rename_features(all_sites)
all_sites.drop(['start','end'],axis=1,inplace=True)

additional_features = ['ATAC','CADD','DANN','Eigen','GenoCanyon','RNASeq','WGBS']
#merge with additional features
with pd.HDFStore(feature_dir+'addtional_features','r') as h5s:
    for feature in additional_features:
        feature_frame = h5s[feature]
        all_sites = pd.concat([all_sites,feature_frame],axis=1)
all_sites = all_sites.loc[:,~all_sites.columns.duplicated()]
all_sites['chr'] = all_sites['chr'].astype('i8')

#nearest tss distance    
chrs = all_sites['chr'].unique()
cols=['chr', 'coordinate','strand']
tss =  pd.read_csv(home+'data/commons/tss.txt',sep='\s+',header=None,names=cols,skiprows=1)
tss = get_winid.convert_chr_to_num(tss,chrs)
tss.sort_values(['chr','coordinate'],inplace=True)
all_sites = nearest_tss(tss,all_sites)

with pd.HDFStore(home+'data/'+dataset+'/all_features','w') as h5s:
    h5s['all_features'] = all_sites