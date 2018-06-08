#! /usr/bin/Rscript
#install.packages("bigmemory",repos='http://cran.us.r-project.org')
library(bigmemory)
source("Script_Appendix.R")
home <- "/home/ec2-user/git/EnsembleCpG/"
extra_storage <- "/home/ec2-user/extra_storage/CpG_EWAS/"
# The user needs to define two vectors: Chr and Pos
# Chr is the chromosome information for all SNPs. Use 23 and 24 to denote chromosome X & Y, respectively. CharToNum function in the appendix can help convert "chr#" -> #.
# Pos is hg19 position for all SNPs.
# Examples (3 SNPs in total): Chr=c(1,2,3), Pos=c(1001, 40000, 12345)
# Given Chr and Pos, the following script can extract GenoCanyon10K scores for all SNPs
dataset = 'RICHS'
file <- paste(home,"data/",dataset,"/all_sites_winid.csv", sep='')
working_dir <- paste(extra_storage,"GenoCanyon/GenoCanyon_10K/",sep='')
output_dir <- paste(extra_storage,"GenoCanyon/Results/",dataset,"/selected_site_scores.txt",sep='')
  
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


