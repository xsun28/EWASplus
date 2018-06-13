source("http://bioconductor.org/biocLite.R")
biocLite("rtracklayer")
library(rtracklayer)
home <- "/Users/Xiaobo/Desktop/"
WGBS <- read.csv(paste(home,"WGBS.csv",sep = ""))
WGBS$chr = paste("chr",WGBS$chr,sep = "")
WGBS$end = WGBS$coordinate + 1
colnames(WGBS) = c("chrom","start","count","end")
ch = import.chain(paste(home,"hg38ToHg19.over.chain",sep = ""))
WGBS = GRanges(WGBS)
seqlevelsStyle(WGBS) = "UCSC" 
new = liftOver(WGBS, ch)
data = as.data.frame(new)
write.csv(data,file = paste(home,"new.csv",sep = ""),col.names = TRUE,row.names = FALSE)