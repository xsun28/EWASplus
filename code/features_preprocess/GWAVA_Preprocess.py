import pandas as pd 
import numpy
from common import commons
home = commons.home
extra_storage = commons.extra_storage
#from features_preprocess import get_winid
import pysam
import pybedtools
import sys, os

class GWAVA_Preprocess(object):
	def __init__(self, data_dir = os.path.join(extra_storage, 'GWAVA'),
				sites_file = os.path.join(home, 'data', 'commons', 'all_sites_winid.csv'),
				additional_feature_file = os.path.join(home, 'data', 'features', 'addtional_features')):
		self.data_dir = data_dir
		self.sites_file = sites_file
		self.additional_feature_file = additional_feature_file

	def process(self):
		"""
		all *_BedTool are 0-indexed
		gwava.bed are sorted lexicographically (chr1, chr10, chr11...)
		we need to use the same logic to treat selected sites file
		"""
		all_sites_df = pd.read_csv(self.sites_file)
		all_sites_df['chr'] = 'chr' + all_sites_df['chr'].astype('str')
		all_sites_df.sort_values(by=['chr', 'coordinate'], inplace=True)
		all_sites_df.rename({'coordinate': 'base_end'}, axis='columns', inplace=True)
		all_sites_df['base_start'] = all_sites_df['base_end'] - 1
		all_sites_df['base_start'] = all_sites_df['base_start'].astype('i8')
		all_sites_df['base_end'] = all_sites_df['base_end'].astype('i8')      
		all_sites_list = [tuple(x) for x in all_sites_df[['chr', 'base_start', 'base_end']].values]
		all_sites_BedTool = pybedtools.BedTool(all_sites_list)
		gwava_file = os.path.join(self.data_dir, 'gwava_scores.bed.gz')
		gwava_BedTool = pybedtools.BedTool(gwava_file)
		closest_gwava_BedTool = all_sites_BedTool.closest(gwava_BedTool, d=True)
		gwava_scores_df = closest_gwava_BedTool.to_dataframe()[['chrom', 'end', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes']]

		gwava_scores_df.columns = ['chr', 'coordinate', 'GWAVA_region_score', 'GWAVA_tss_score', 'GWAVA_unmatched_score', 'GWAVA_dist_to_nearest_snp'] #1-based index here
		gwava_scores_df['chr'] = gwava_scores_df['chr'].str.replace('chr', '').astype('int') #be consistent with other additional features

		agg_dict = {'GWAVA_region_score': 'mean',
					'GWAVA_tss_score': 'mean',
					'GWAVA_unmatched_score': 'mean',
					'GWAVA_dist_to_nearest_snp': 'mean'
		}
		gwava_scores_df = gwava_scores_df.groupby(['chr', 'coordinate']).agg(agg_dict).reset_index()
		with pd.HDFStore(self.additional_feature_file,'a') as h5s:
			h5s['GWAVA'] = pd.DataFrame(gwava_scores_df) 
