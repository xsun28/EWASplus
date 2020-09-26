# EWASplus pipeline
This repository contains all components of the pipeline for predicting novel Alzheimer's Disease (AD)-associated CpG sites beyond 450K array across the whole human genome, including experimental set construction, features collection/processing, features selection and ensemble learning, for each of the AD-associated trait of interest. 

## Tools
* Python 3.6
* R 3.4
* Amazon Elastic Compute Cloud (AWS EC2)

## Prerequites
The following input files are needed. Inputs files marked as optional are only needed if the user would like to reproduce the raw data processing procedure from scratch. The recommended way is to use the processed features matrix for 26M CpG loci directly (i.e., starts from step 3 in "Running the pipeline" section) since the raw data processing requires significant time and hardware configuration.
* CSV files with EWAS differential methylation summary level data based on 717 participants from ROS/MAP cohort. For each trait, the file includes CpG ID, F statistics and p-values (null hypothesis: AD samples have the same methylation level as control samples) for *CpG sites whose methylation level was measured using Illumina 450K array (i.e. 450K sites)*. `ROSMAP.csv`
* a TXT file with the whole human genome spread across 200 base-pair intervals `wins.txt`
* (optional) BED files with window IDs of *all CpG sites across the whole human genome (i.e. WGBS sites)* and values of the 1806 features used in our previously published work on DIVAN.  `DIVAN_features.bed`
* (optional) TSV.GZ files with genomic locations of WGBS sites and CADD scores `CADD.tsv.gz`
* (optional) TSV.BGZ files with genomic locations of WGBS sites and DANN scores `DANN.tsv.bgz`
* (optional) TAB.BGZ files with genomic locations of WGBS sites and EIGEN scores `EIGEN.tab.bgz`
* (optional) BED.GZ files with genomic locations of WGBS sites and GWAVA scores `GWAVA.bed.gz`
* (optional) BED files with window IDs of WGBS sites and RNA-sequencing read counts data `RNASEQ.bed`
* (optional) BED files with window IDs of WGBS sites and ATAC-sequencing read counts data `ATACSEQ.bed`
* (optional) BED files with genomic locations of WGBS sites and WGBS read counts data `wgbs_readcounts.bed`
* (optional) a TXT file with genomic locations of transcription start sites (tss) `tss.txt`



## Running the pipeline 

**0) Preliminary step: hg38 to hg19 conversion**

The genomic coordinates in this pipeline are 1-indexed/in hg19. The original ENCODE WGBS datasets are 0-indexed/in hg38 and therefore need to be converted. This conversion can be completed by running the `WGBS_allsites_preprocess`.py script. The number of studied WGBS sites is reduced from `approximatly 28 million` to `approximatly 26 million` after LiftOver conversion due to multiple reasons (inconsistent chromosome, one-to-many conversion results, etc). 

``` 
python prediction/WGBS_allsites_preprocess.py
```
The file `${wgbs_readcounts.bed}`contains the 0-indexed/hg38 genomic locations of WGBS sites.

This step generates `all_wgbs_sites_winid.csv`, which contains the genomic locations in both hg38 and hg19 and window IDs for WGBS sites. 

**1) Asssign feature values to WGBS sites**

To prepare for future prediction of AD-associated WGBS sites, we first assign all (N=2256) feature values to WGBS sites. 

By running `WGBS_all_sites_feature_preprocess.py`, the features for CpG loci of the whole genome are assigned in 14 batches (batch size = 2 millions). The number of batches is highly customized considering the RAM limit.

``` 
python features_preprocess/WGBS_all_sites_feature_preprocess.py
```

The features are processed as follows:

* 1806 read counts features from various genomic profiles adopted in a previous related study [DIVAN: accurate identification of non-coding disease-specific risk variants using multi-omics profiles](https://link.springer.com/article/10.1186/s13059-016-1112-z) are constructed to cover the entire human genome in 200 base-pair resolution and are assigned to each CpG locus by matching the window ID
* CADD, DANN, EIGEN, GenoCanyon and GWAVA scores are assigned to each CpG locus by matching genomic location
* RNA-sequencing, ATAC-sequencing and WGBS read counts data are assigned to each CpG locus by matching window ID
* Distance to the nearest tss is calculated for each CpG locus
* Summary of all features 

  | Feature source         | Number   | 
  | -------------          |:--------:| 
  | REMC DNase             | 73       |
  | REMC Histone           | 735      | 
  | ENCODE DNase           | 80       |   
  | ENCODE FAIRE           | 31       |   
  | ENCODE TF(HAIB)        | 292      |   
  | ENCODE TF(SYDH)        | 279      |   
  | ENCODE Histone         | 267      |   
  | ENCODE RNA Polymerase  | 49       |   
  | ENCODE RNA-seq         | 243      |   
  | ENCODE ATAC-seq        | 66       | 
  | ENCODE WGBS            | 127      |  
  | GenoCaynon             | 1        |   
  | Eigen                  | 4        |   
  | DANN                   | 2        |   
  | CADD                   | 2        |   
  | GWAVA                  | 4        | 
  | Distance to nearest TSS| 1        | 
  | Total                  | 2256     | 
  
  


This step generates HDF5 files with matched features for all CpG loci across the human genome (N=26573858):
``` 
all_features_0_2000000.h5
all_features_2000000_4000000.h5
....
```

**2) Assign feature values to all 450K sites**

By running the all450k_feature_preprocess.py script, the feature values for all CpG loci on 450K array are assigned.

``` 
python features_preprocess/all450k_feature_preprocess.py
```
This step generates a HDF5 file for 450K sites and their feature values:
``` 
all_450k_features.h5
```

**3) Experimental set construction for each trait**

For furture model training purpose, the experimental set is selected for each trait based on EWAS summary statistics. For each AD related trait, we include positive sites (signficantly associated with AD) and negative sites (not significantly associated with AD). The sites selection criteria are as follows:

a) Select positive sites whose p-values are below trait-specific threshold

b) For each selected positive site, select 10 matching negative sites that:

* have p-values greater than 0.4
* have the same methylation status (either hyper- and hypo-) as the corresponding positive site
* have the closest β-values as the positive site 

``` 
python sites_selection/AD_sites_selection.py     
```

This step outputs multiple CSV files for different traits: 
``` 
all_sites_winid.csv
```
which contains the selected experimental set with columns: CpG ID, chromosome, coordinate, p-value, β-value, label (0 for negative sites/ 1 for positive sites) and window ID. 

and one CSV file for all 450K sites:
``` 
all_450k_sites_winid.csv
```
which contains all 450k sites with columns: CpG ID, chromosome, coordinate, p-value, β-value and window ID. 


**4) Assign feature values to the experimental set for each trait**

By running the all_features_preprocess.py script, the feature values for the selected training set are assigned.

``` 
python features_preprocess/all_features_preprocess.py
```


This step generates multiple HDF5 files for different traits:
``` 
all_features.h5
```
which contains all feature values of the experimental set for each trait.


**5) Feature selection for each trait**

Considering the number of features is greater than the number of CpG sites in the experimental set, features selection is performed for each trait to alleviate overfitting. 

The feature selection process is achived by running the feature_selection.py script. 

``` 
python features_selection/feature_selection.py 
```
The features are selected as follows for each trait: 

* Split training/testing data on 9:1 ratio 
* Select top 100 significant features by fitting the training data using random forest, xgboost, logistic regression and support vector classifier (SVC) with linear kernel, respectively. 
* For each feature, summarize the number of classifiers that select it, n (0≤n≤4) 
* Keep features with n≥2
* For each of the preserved feature, perform Wilcoxon rank-sum test and calculate the p-values under the null hypothesis: AD associated CpG loci have the same feature values as non-AD associated CpG loci
* From top to bottom, sort selected features first by desceding n and then by ascending p-value 
* Select top ranked features for each trait 

This step outputs a CSV file and a HDF5 file for each trait:

1)`feature_stats.csv`, which contains information of the top ranked feaures, including feature name, p-value, and n

2)`selected_features.h5`, which contains the training and testing set with the values of top ranked features assigned and the labels for training and testing set.

**6) Model hyper-parameters tuning and model selection for each trait**

Four candidate base learners including random forest, xgboost, logistic regression with L2-regularization and SVC with linear kernel are involved in this step. The optimal  hyper-paramters for each base learner and the best combination of base base leaner are selected by running the ModelSelectionTuning.py script.

``` 
python models/ModelSelectionTuning.py 
```

The optimal hyper-parameters for each base classifier and and the best combination of base classifiers are selected as follows:

* Use the training set (generated from step 5) with 3-fold cross-validation to select the optimal hyper-parameters for each base classifier
* Use the training/testing set (generated from step 5) with 10-fold cross-validation to evaluate all possible combinations of base classifiers
* Calculate average AUC, F1-score, etc across all 10 folds 
* Select the best combination of base classifiers 


**7) Inference for the whole genome trait-specific AD risky scores**

Infer the AD risky scores for all CpGs across the whole human genome with the optimial base learner combination from step 6:

``` 
python prediction/WGBS_prediction.py
```

In this step, we need to retrain the base classifiers in the ensemble model with the entire experimental set to obtain the optimal hyper-parameters and save the retrained model. However if the retrain process has already been done, we load the saved retrained model directly. Then, we use the trained model to predict the probabilities of being associated with AD for whole genome CpG loci. CpG loci are ranked in the descendig order for prioritization.

This step generates 2 CSV files, a HDF5 file and a pkl file for each trait:

1)`pred_positive_500.csv`, which contains the top 500 sites and their probabilities of being positive; 

2)`top500_nearest_450k.csv`, which contains the 450K sites within 5k up/downstream of top 500 predicted sites with their CpG ID and genomic location;

3)`pred_probs.h5`, which contain all CpG loci and their probabilities of being positive for all traits; 

4)the saved retrained model in `prediction_model.pkl`;

and a HDF5 file `pred_probs_450k.h5`, which contains all 450K sites and their probabilities of being positive,


**8) Combine results for all AD traits**

To assess the AD risky scores generated from each independently trained model, both the average and the weighted average of all predicted probabilities are calculated. The weights for each trait is determined based on the F1 score. 

``` 
python prediction/WGBS_alltraits_prediction_AD.py 
```

This step generates 2 CSV files for WGBS sites:

1)`common_top500_mean_nearest_450k.csv`, which contains the top 500 sites with highest unweighted average of 7 probabilities, and their distances to the nearest tss; 

2)`common_top500_weighted_nearest_450k.csv`,  which contains the top 500 sites with highest weighted average of 7 probabilities, and their distances to the nearest tss; 

and a csv file for 450K sites:
`450kwithpredictedprob.csv`, which contains the unweighted and weighted average of 7 probabilities of all 450K sites. 
