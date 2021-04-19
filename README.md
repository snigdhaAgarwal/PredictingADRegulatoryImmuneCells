# PredictingADRegulatoryImmuneCells
Predicting regulatory activity of AD SNPs in Immune Cell types. Masters Thesis work done at Pfenning Lab @ CMU. Supervised by Andreas Pfenning, Easwaran Ramamurthy.

## Files Guide

**cnn.py**
Was used to train classification models

**nullSet.R**
Used to generate GC matched negatives using genNullSeqs. Usage: Rscript nullSet.R hg19 *bed_file_name_without_the_extension*

**signal_extraction.py**
Contains code for generating one hot encoded sequences along with adding reverse complements, grouping Satpathy defined clusters into 8 immune cell clusters, converting called peak bed file to npy files for model trainig and other intermediate data processing methods. 

## Contact Information
Snigdha Agarwal 
snigdhaagarwal93@gmail.com
