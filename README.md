# PredictingADRegulatoryImmuneCells
Predicting regulatory activity of AD SNPs in Immune Cell types. Masters Thesis work done at Pfenning Lab @ CMU. Supervised by Andreas Pfenning, Easwaran Ramamurthy.

## Section code guide

### Single Label Regression

#### Reads per million normalization (signal_extraction.py)
saveClusterIndices(): To get the total read counts from the matrix file, we first get indices/row number of each cell in a cluster. The indices would tell us which value in the 2nd column  of the matrix file to access. 

generateSignals(): Converts the sparse matrix to a peak by cell matrix populated by read counts. The schematic below shows such a matrix where the colors are different clusters. To get a peak's signal in a cluster, sum up the read counts for all the cells in that cluster. To get RPM, multiply with a million and divide by the total reads for all the cells in the cluster.

![Screen Shot 2021-06-29 at 10 12 55 PM](https://user-images.githubusercontent.com/59067635/123891611-299c6780-d927-11eb-930f-92035a470006.png)
  

## Data Preview

### Barcodes file
Contains the cluster information according to Satpathy group for each cell
![Screen Shot 2021-06-29 at 9 55 59 PM](https://user-images.githubusercontent.com/59067635/123890180-cc9fb200-d924-11eb-8fdf-78d406dfcf12.png)

### Matrix file
Sparse matrix where first column is peak indices, second column is cell indices and third column is number of reads for that peak in that cell
![Screen Shot 2021-06-29 at 9 58 29 PM](https://user-images.githubusercontent.com/59067635/123890406-256f4a80-d925-11eb-8009-60e7ff4513fb.png)



## Files Guide

**cnn.py**
Was used to train single-label and multilabel classification models

**nullSet.R**
Used to generate GC matched negatives using genNullSeqs. Usage: Rscript nullSet.R hg19 *bed_file_name_without_the_extension*

**signal_extraction.py**
Contains code for generating one hot encoded sequences along with adding reverse complements, grouping Satpathy defined clusters into 8 immune cell clusters, converting called peak bed file to npy files for model training and other intermediate data processing methods. 

**tsv_extract.sh**
Converts raw fragment files from Satpathy paper to bam file format required by SCATE, for a single cell type. Need to change file names and paths to run for each of the 8 cell types. The input cell barcodes for each cell type were filtered using the command: ![image](https://user-images.githubusercontent.com/1850984/115171091-43504180-a090-11eb-9d62-de95285c49fe.png)

**myScate.R**
Runs the SCATE process to generate peaks and signal for a single cell type. Need to change file names and paths to run for each of the 8 cell types. This code was run with SLURM config : \
\#SBATCH --mem=40G \
\#SBATCH -c 1

**regression.py**
Training single label regression models for both SCATE and ArchR peaks. Usage: python3 regression.py --xtrain Archr/mono/Cluster3_trainInput.npy --ytrain Archr/mono/Cluster3_trainLabels.npy --xvalid Archr/mono/Cluster3_validationInput.npy --yvalid Archr/mono/Cluster3_validationLabels.npy --model-out output-newDrop.hdf5  -md train -d Archr/mono/

**haplo.R**
Used to get SNPs in LD with lead AD SNPs, using HaploReg tool.

**tanzi_variants_construct_sequence_and_propagate.py**
To score SNPs using trained models. Usage: ![image](https://user-images.githubusercontent.com/1850984/115271012-e6de3800-a10a-11eb-8312-560f12244165.png)\
-s: List of SNPs. Format: \
![Screen Shot 2021-04-19 at 12 31 42 PM](https://user-images.githubusercontent.com/1850984/115271316-33297800-a10b-11eb-9743-bb60cace294c.png)
-o: Overlap of your peaks with above SNPs. Same length as SNP file. Last column is 0 if no overlap and any number if there is an overlap. Can generate this file using bedtools intersect -c. Format: \
![Screen Shot 2021-04-19 at 12 33 39 PM](https://user-images.githubusercontent.com/1850984/115271566-797ed700-a10b-11eb-8f5f-f067cff1555f.png)\
-m : Model hdf5 file \
-left: Number of nucleotides to add on left of SNP \
-right: Number of nucleotides to add on right of SNP

**checkVariants.ipynb**
Used to identify AD SNPs that impact regulatory activity in each cell type. Also used to see correlation of Tanzi SNP scores as predicted by Easwaran's bulk models, SCATE peaks and Archr peaks. 

## Contact Information
Snigdha Agarwal 
snigdhaagarwal93@gmail.com
