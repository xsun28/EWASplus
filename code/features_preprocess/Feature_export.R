#! /usr/bin/Rscript
# pos_winid<-read.table('/Users/Xiaobo/Jobs/CpG/pos_winid.csv',header =FALSE)
# neg_winid<-read.table('/Users/Xiaobo/Jobs/CpG/neg_winid.csv',header=FALSE)
#home <- '/home/ec2-user/git/EnsembleCpG/data/'
args = commandArgs(trailingOnly=TRUE)
home <- paste(args[1],'/',sep='')
dataset = paste(args[2],'/',sep='')
is_450k <- as.logical(args[3])
tablename <- 'selected_450k_pos_winid.csv' if(is_450k) else 'selected_pos_winid.csv'
filesuffix <- '_all_450k.csv' if(is_450k) else '_all.csv'
    
all_sites_winid<-read.table(paste(home,dataset,tablename,sep=''),header=FALSE)
dir <- paste(home,'features/',sep='')

files <- list.files(dir,".*adjust.rda$")
dir.create(paste(dir,dataset,sep=''),showWarnings = FALSE)
for (file in files){
  load(paste(dir,file,sep=''))
  #pos <- readmat[unlist(pos_winid),]
  #neg <- readmat[unlist(neg_winid),]
  all <- readmat[unlist(all_sites_winid),]
  file_name <- unlist(strsplit(file,'\\.'))[1]
  #write.csv(pos,file=paste(dir,file_name,'_Pos.csv',sep=''),row.names=F)
  #write.csv(neg,file=paste(dir,file_name,'_neg.csv',sep=''),row.names=F)
  write.csv(all,file=paste(dir,dataset,file_name,filesuffix,sep=''),row.names=F)
  #rm(readmat,pos,neg)
  rm(readmat,all)
}


# load('/Users/Xiaobo/Jobs/CpG/data/features/Duke_DNaseI_HS.Reads.db.adjust.rda')
# #pos <- readmat[unlist(pos_winid),]
# #neg <- readmat[unlist(neg_winid),]
# all <- readmat[unlist(all_sites_winid),]
# 
# #write.csv(pos,file=paste(dir,'Duke_DNaseI_HS_Pos.csv',sep=''),row.names=F)
# #write.csv(neg,file=paste(dir,'Duke_DNaseI_HS_neg.csv',sep=''),row.names=F)
# write.csv(all,file=paste(dir,'Duke_DNaseI_HS_all.csv',sep=''),row.names=F)
# #rm(readmat,pos,neg)
# rm(readmat,all)
###################################


# load('/Users/Xiaobo/Jobs/CpG/data/features/Roadmap_DNase.Reads.db.adjust.rda')
# #pos <- readmat[unlist(pos_winid),]
# #neg <- readmat[unlist(neg_winid),]
# all <- readmat[unlist(all_sites_winid),]
# 
# write.csv(pos,file=paste(dir,'Roadmap_DNase_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'Roadmap_DNase_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)

###################################

# load('/Users/Xiaobo/Jobs/CpG/data/features/Roadmap_Histone.Reads.db.adjust.rda')
# pos <- readmat[unlist(pos_winid),]
# neg <- readmat[unlist(neg_winid),]
# all <- readmat[unlist(all_sites_winid),]
# write.csv(pos,file=paste(dir,'Roadmap_Histone_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'Roadmap_Histone_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)


###################################

# load('/Users/Xiaobo/Jobs/CpG/data/features/UNC_FAIRE.Reads.db.adjust.rda')
# pos <- readmat[unlist(pos_winid),]
# neg <- readmat[unlist(neg_winid),]
# write.csv(pos,file=paste(dir,'UNC_FAIRE_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'UNC_FAIRE_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)


###################################

# load('/Users/Xiaobo/Jobs/CpG/data/features/wgEncodeBroadHistone.Reads.db.adjust.rda')
# pos <- readmat[unlist(pos_winid),]
# neg <- readmat[unlist(neg_winid),]
# write.csv(pos,file=paste(dir,'wgEncodeBroadHistone_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'wgEncodeBroadHistone_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)


###################################

# load('/Users/Xiaobo/Jobs/CpG/data/features/wgEncodeHaibTfbs.Reads.db.adjust.rda')
# pos <- readmat[unlist(pos_winid),]
# neg <- readmat[unlist(neg_winid),]
# write.csv(pos,file=paste(dir,'wgEncodeHaibTfbs_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'wgEncodeHaibTfbs_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)

###################################

# load('/Users/Xiaobo/Jobs/CpG/data/features/wgEncodeSydhTfbs.Reads.db.adjust.rda')
# pos <- readmat[unlist(pos_winid),]
# neg <- readmat[unlist(neg_winid),]
# write.csv(pos,file=paste(dir,'wgEncodeSydhTfbs_Pos.csv',sep=''),row.names=F)
# write.csv(neg,file=paste(dir,'wgEncodeSydhTfbs_neg.csv',sep=''),row.names=F)
# rm(readmat,pos,neg)


