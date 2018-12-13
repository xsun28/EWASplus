#running script

from common import commons
home = commons.home
import pandas as pd
import numpy as np
from common import Convert_To_Normal_Dist as ctnd
from features_selection import TTest as tt
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from hyperopt import fmin,tpe,hp, STATUS_OK
import math
from features_selection import TTest, WilcoxonRankSums


#--------------------------------------------------------------------- 
def normalize_feature(all_features,dataset='AD_CpG'):
    count_thresh = 10
    unique_value_counts = [len(all_features[col].unique()) for col in all_features.columns[2:]] #ignore chr and coordinate
    normalize_all_features = all_features.copy()
    mappings = pd.DataFrame()
    for col,num in zip(all_features.columns[2:],unique_value_counts):
        if num >= count_thresh:
            print(col)
            atn = ctnd.AnyToNormal(col)
            mapping = atn.transform(normalize_all_features)
            mappings[mapping.columns[0]] = mapping.ix[:,0]
            mappings[mapping.columns[1]] = mapping.ix[:,1]
        else: 
            continue
    h5s = pd.HDFStore(home+'data/'+dataset+'/normalize_mapping','w')
    h5s['mappings'] = mappings
    h5s['normal_features'] = normalize_all_features
    h5s.close()
    return home+'data/'+dataset+'/normalize_mapping',normalize_all_features,mappings
   
#-------------------------------------------------------------------------------
def data_selection(data,classes=[0,1,-1],combine=False,*args):              #which classes of data to keep
    if not isinstance(data,pd.DataFrame):
        cols = args
        data = pd.DataFrame(data,columns=cols)    
    ret_data = data.query('label in @classes')
    for i,label in enumerate(classes):
        ret_data['label'][ret_data['label']==label] = i
    
    if len(classes)>2 and combine==True:
        ret_data['label'] = ret_data['label'].where(ret_data['label']==0,1)
    return ret_data
        
#------------------------------------------------------------------------------                  
 
def TSNEPlot(data,class_labels,param_map):##param_map like 1:('r','o','80')
    data1 = data.reset_index()
    tsne = TSNE(n_components=3,random_state=91)
    feature_reduced = tsne.fit_transform(data1.drop(['label','index'],axis=1))
    fig = plt.figure(figsize=(18,15))
    ax = fig.gca(projection='3d')
    for i in class_labels:
        c,marker,size = param_map[i]
        print(c,marker,size)
        index = data1.query('label == @i').index.values
        ax.scatter(feature_reduced[index,0],feature_reduced[index,1],feature_reduced[index,2],c=c,marker=marker,s=size)
    plt.show()
    return None

#-------------------------------------------------------------------------------
def simulate(data,label,**argkw):
    data = data[data['label']==label]
    encoder = AS.dataset_simulator(**argkw)
    encoder.fit(np.array(data.drop('label',axis=1)))
    sim_data = pd.DataFrame(encoder.transform(),columns=data.columns.drop('label'))
    sim_data['label'] = label
    return sim_data
          
#-----------------------------------------------------------------------------
def subset_control(data,num):      ## randomly select a subset of certain ratio positive vs control samples 
    pos_index = data.query('label!=0').index.values
    control_sub_index = np.random.permutation(data.query('label==0').index.values)[:len(pos_index)*num]
    select_index = np.concatenate((pos_index,control_sub_index))
    return data.loc[select_index,:]
#------------------------------------------------------------------------------
def sparse_autoencoder_score(ae_params):
    global encoders
    sparse_ae = sac.sparse_autoencoder(**ae_params)
    score = commons.cross_validate_score(sparse_ae,reduced_train_x,train_label,ae_sample_weights_train)
    encoders.extend([sparse_ae])
    if math.isnan(score) or math.isfinite(score):
        score = np.Infinity
    return {'loss':-score,'status':STATUS_OK}
#-----------------------------------------------------------------
def selected_feature_analysis(features,X,y):
    stats = []
    pos = X[y==1]
    neg = X[y==0]
    for feature in features:
        #test = TTest.FeatureTTest(feature)
        test = WilcoxonRankSums.Ranksums(feature)
        stats.extend([test.fit(pos,neg)])
    return pd.DataFrame(stats)
#--------------------------------------------------------------------
def method_params(methods=['random_forest','xgboost','logistic_regression','linear_SVC']):
    params={}
    feature_num = train_x.shape[1]
    #class_weight = {0:1,1:30}
    class_weight = None
    l_param={'C':9,'penalty':'l1'}
    rf_param = {'n_estimators':1000,'max_depth':10,'min_samples_split':14 ,'min_samples_leaf':1, 'n_jobs':-1 }
    svc_param = {'C':2,'dual':False,'penalty':'l1'}
    xgb_param = {'learning_rate':0.1,'max_depth':10,'n_estimators':1500,'reg_lambda':40,'gamma':1,'n_jobs':-1}
    mutual_information_param = {'k':100}
    fisher_param = {'k':100}
    if 'logistic_regression' in methods:
        params['logistic_regression'] = l_param
    if 'random_forest' in methods:
        params['random_forest'] = rf_param
    if 'linear_SVC' in methods:
        params['linear_SVC'] = svc_param
    if 'xgboost' in methods:
        params['xgboost'] = xgb_param
    if 'mutual_information' in methods:
        params['mutual_information'] = mutual_information_param
    if 'fisher_score' in methods:
        params['fisher_score'] = fisher_param   
    return params