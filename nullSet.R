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
genNullSeqs(paste(args[2],'.bed',sep=""), #"combined2pos.bed",'2_hg19.bed',
  genome=genome,
  outputBedFN = paste(args[2],'-GCneg.bed',sep=""),#'combined2GC-neg.bed','test-neg.bed',
  # outputPosFastaFN = paste(args[2],'.fa',sep=""),#'combined2pos.fa','test.fa',
  # outputNegFastaFN = paste(args[2],'-neg.fa',sep=""),#'neg2GC-combined.fa','test-neg.fa',
  length_match_tol = 0, # want exact same length negatives
  xfold = 1, # twice the number of negatives
	nMaxTrials=180)
