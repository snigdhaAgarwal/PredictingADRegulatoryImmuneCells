#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
library(gkmSVM)
if (args[1]=='hg19') {
  library(BSgenome.Hsapiens.UCSC.hg19.masked)
  genome <- BSgenome.Hsapiens.UCSC.hg19.masked
} else {
  library(BSgenome.Hsapiens.UCSC.hg38.masked)
  genome <- BSgenome.Hsapiens.UCSC.hg38.masked
}
print(args)
genNullSeqs(paste(args[2],'.bed',sep=""), 
  genome=genome,
  outputBedFN = paste(args[2],'-GCneg.bed',sep=""),
  length_match_tol = 0, # want exact same length negatives
  xfold = 2, # twice the number of negatives
	nMaxTrials=180)
