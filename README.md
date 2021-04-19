# PredictingADRegulatoryImmuneCells
Predicting regulatory activity of AD SNPs in Immune Cell types. Masters Thesis work done at Pfenning Lab @ CMU. Supervised by Andreas Pfenning, Easwaran Ramamurthy.

## Files Guide

**cnn.py**
Was used to train single-label and multilabel classification models

**nullSet.R**
Used to generate GC matched negatives using genNullSeqs. Usage: Rscript nullSet.R hg19 *bed_file_name_without_the_extension*

**signal_extraction.py**
Contains code for generating one hot encoded sequences along with adding reverse complements, grouping Satpathy defined clusters into 8 immune cell clusters, converting called peak bed file to npy files for model training and other intermediate data processing methods. 

**tsv_extract.sh**
Converts raw fragment files from Satpathy paper to bam file format required by SCATE, for a single cell type. Need to change file names and paths to run for each of the 8 cell types. The input barcodes were filtered using the command: ![image](https://user-images.githubusercontent.com/1850984/115171091-43504180-a090-11eb-9d62-de95285c49fe.png)

**myScate.R**
Runs the SCATE process to generate peaks and signal for a single cell type. Need to change file names and paths to run for each of the 8 cell types. This code was run with SLURM config : \
\#SBATCH --mem=40G \
\#SBATCH -c 1

**regression.py**
Training single label regression models for both SCATE and ArchR peaks. Usage: python3 regression.py --xtrain Archr/mono/Cluster3_trainInput.npy --ytrain Archr/mono/Cluster3_trainLabels.npy --xvalid Archr/mono/Cluster3_validationInput.npy --yvalid Archr/mono/Cluster3_validationLabels.npy --model-out output-newDrop.hdf5  -md train -d Archr/mono/

**haplo.R**
Used to get SNPs in LD with lead AD SNPs, using HaploReg tool.

## Contact Information
Snigdha Agarwal 
snigdhaagarwal93@gmail.com
