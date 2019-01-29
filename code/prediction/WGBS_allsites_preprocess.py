##running scripts: Get all WGBS sites hg19 coordinate and window id
import sys
from common import commons
home = commons.home
extra_storage = commons.extra_storage
import pandas as pd
from features_preprocess import get_winid
import numpy as np
import re
from pyliftover import LiftOver
logger = commons.logger
from features_preprocess.get_winid import convert_num_to_chrstr

def read_WGBS(file):
    bed = pd.read_csv(file,usecols=[0,1,2,5,9,10],header=None,names=['chr','pos1','pos2','strand','total','percent'],sep='\s+')
    bed['coordinate'] = np.where(bed['strand']=='+',bed['pos1'],bed['pos1']-1)  ##read 0-based WGBS bed, merge +/- strand
    bed.drop(['pos1','pos2'],axis=1,inplace=True)
    bed['count'] = np.round(bed['total']*bed['percent']/100.0)
    bed.drop(['total','percent'],axis=1,inplace=True)
    bed = bed.groupby(['chr','coordinate']).aggregate({'count':sum}).reset_index()
    
    #    bed_counts = bed.groupby(['chr','coordinate']).aggregate({'count':sum})
    return bed

def hg38tohg19(row):
    global lo
    hg19 = lo.convert_coordinate('chr'+convert_num_to_chrstr(row[1]['chr']),row[1]['start'])
    if(len(hg19)>0):
        Chr,start,strand,score = hg19[0]
        try:
            Chr = int(Chr[3:])
        except ValueError:
            if Chr[3:] == 'X':
                Chr = 23
            elif Chr[3:] == 'Y':
                Chr = 24
        start = start+1 ##convert to 1-bbased WGBS coordinate
        end = start
        return [Chr,start,end,row[1]['chr'],row[1]['start']]
    else:
        return [None,None,None,row[1]['chr'],row[1]['start']]
    
    
    
logger.info('starting preprocess all WGBS sites from hg38 to hg19, and obtain window for each WGBS site for merging 1806 features later...')
dataset = 'WGBS'
win_path= home+'data/commons/wins.txt'
chrs=np.arange(1,25,dtype='int64')
wins = get_winid.read_wins(win_path,chrs)
all_wgbs_sites_file = home+'data/'+dataset+'/all_wgbs_sites_winid.csv'
hg19_wgbs_file = home+'data/'+dataset+'/hg19_WGBS.csv'

###get all WGBS sites only need to run once
#data_dir = extra_storage+'WGBS/'
#file = data_dir+'ENCFF844EFX.bed'
#wgbs_file = home+'data/'+dataset+'/WGBS.bed'
#bed = read_WGBS(file)
#bed = get_winid.convert_chr_to_num(bed,chrs).sort_values(['chr','coordinate'])
#bed.rename({'coordinate':'start'},axis=1,inplace=True)
#bed['end'] = bed['start']+1
#bed.drop(['count'],axis=1,inplace=True)
#bed.to_csv(wgbs_file,columns=['chr','start','end'],index=False,sep="\t")


###convert to hg19, only need run once
#lo = LiftOver('hg38', 'hg19')
#coord_hg19 = [hg38tohg19(row)for row in bed.iterrows()]
#coord_hg19 = pd.DataFrame(coord_hg19,columns=['chr','coordinate','end','hg38chr','hg38coordinate']).query('chr in @chrs')
#coord_hg19.dropna().drop_duplicates(['chr','coordinate']).to_csv(hg19_wgbs_file,index=False)

#using WGBS(hg19) sites only run once
logger.info('reading hg19/hg38 wgbs files from '+hg19_wgbs_file)
hg19_wgbs = pd.read_csv(hg19_wgbs_file,usecols=[0,1,3,4]).sort_values(['hg38chr','hg38coordinate']).reset_index(drop=True)
#hg19_wgbs = get_winid.convert_chr_to_num(hg19_wgbs,chrs)
logger.info('Obtaining winid of all hg19 wgbs sites...')
all_sites = get_winid.get_winid(wins,hg19_wgbs,True).dropna()
all_sites['winid'] = all_sites['winid'].astype('i8')
logger.info('Saving all WGBS sites with window id to '+all_wgbs_sites_file)
all_sites.to_csv(all_wgbs_sites_file,index=False)