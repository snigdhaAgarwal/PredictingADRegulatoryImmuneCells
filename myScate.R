if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("GenomicAlignments","preprocessCore"))
if (!require("devtools"))
  install.packages("devtools")
install.packages("mclust",repos="https://CRAN.R-project.org")
install.packages("xfun",repos="https://CRAN.R-project.org")
install.packages(c("Rcpp","splines2","xgboost","RcppArmadillo","data.table"),repos="https://CRAN.R-project.org")

Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")
devtools::install_github("zji90/SCATE",repos = c("https://CRAN.R-project.org",BiocManager::repositories()), verbose=TRUE)

library(SCATE)
#
satacprocess1 <- function(input,type='bam',libsizefilter=1000) {
      if (type=='bam') {
        my_list <- sapply(input,readGAlignments)
        satac <- sapply(my_list,GRanges)
      } else {
            satac <- input
      }
      satac <- satac[sapply(satac,length) >= libsizefilter]
      n <- names(satac)
      satac <- lapply(satac,function(i) {
            start(i) <- end(i) <- round((start(i) + end(i))/2)
            i
      })
      names(satac) <- n
      satac
}

cur_path = "scate_files/progen/bam_files"

bamlist <- list.files(path = cur_path,pattern = "\\.bam$",full.names=TRUE)
satac <- satacprocess1(input=bamlist,type='bam')
length(satac)

usercellcluster <- rep(1,each=length(satac))
names(usercellcluster) <- names(satac)
res <- SCATE(satac,genome="hg19",cluster=usercellcluster,clusterid=NULL,clunum=NULL,ncores=1,verbose=TRUE)
length(res)
res
# add fdrcut only if want more number of peaks for downstream analysis
peakres <- peakcall(res,fdrcut=0.05)
head(peakres[[1]])
length(peakres[[1]])
write.table(peakres[[1]],file='scate_files/progen/big_peaks.bed',sep='\t',quote=F,col.names=F,row.names=F)
