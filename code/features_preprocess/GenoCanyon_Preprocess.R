#! /usr/bin/Rscript
#install.packages("bigmemory",repos='http://cran.us.r-project.org')
library(bigmemory)
library('methods')
source("Script_Appendix.R")
args = commandArgs(trailingOnly=TRUE)
is450k <- as.logical(args[1])
home <- paste(args[2],'/',sep='')
extra_storage <- args[3]
dataset <- args[4]

#is450k <- FALSE
#home <- "/home/ec2-user/volume/git/EnsembleCpG/"
#extra_storage <- "/home/ec2-user/volume/git/EnsembleCpG/data/raw/"
# The user needs to define two vectors: Chr and Pos
# Chr is the chromosome information for all SNPs. Use 23 and 24 to denote chromosome X & Y, respectively. CharToNum function in the appendix can help convert "chr#" -> #.
# Pos is hg19 position for all SNPs.
# Examples (3 SNPs in total): Chr=c(1,2,3), Pos=c(1001, 40000, 12345)
# Given Chr and Pos, the following script can extract GenoCanyon10K scores for all SNPs
#dataset = 'AD_CpG/amyloidwith'
if (!is450k){
    file <- paste(home,"data/",dataset,"/all_sites_winid.csv", sep='')
}else{
    file <- paste(home,"data/",dataset,"/all_450k_sites_winid.csv", sep='')
}
working_dir <- paste(extra_storage,"GenoCanyon/GenoCanyon_10K/",sep='')
dir.create(paste(extra_storage,"GenoCanyon/Results/",dataset,sep=''),showWarnings = FALSE)
if (!is450k){
    output_dir <- paste(extra_storage,"GenoCanyon/Results/",dataset,"/selected_site_scores.txt",sep='')
}else{
    output_dir <- paste(extra_storage,"GenoCanyon/Results/",dataset,"/selected_site_all_450k_scores.txt",sep='')
}  
all_sites <- read.table(file,header=TRUE,sep=',')
Chr <- c(all_sites[['chr']])
Pos <- c(all_sites[['coordinate']])


GenoCanyon = rep(NA, length(Chr))
setwd(working_dir)
pb = txtProgressBar(0, 45, style=3)
for(i in 1:45){
  Region = as.logical((Chr == Chr.GenoCanyon10K[i])*(Pos %in% (PosStart.GenoCanyon10K[i]:PosStop.GenoCanyon10K[i])))
  if(sum(Region) > 0){
    mat = attach.big.matrix(files.GenoCanyon10K[i])
    GenoCanyon[Region] = mat[Pos[Region] - PosStart.GenoCanyon10K[i] + 1]
  }
  setTxtProgressBar(pb, i)
}

# Output GenoCanyon10K scores
write.table(GenoCanyon, output_dir, quote=F, row.names=F, col.names=F)


