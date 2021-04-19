#!/bin/bash

# a[$7] in awk means an array with index as string that is in $7. Storing $3 as
# value allows us to print cluster name later by checking 4th field of tsv file
# against index names in a

#complete flow for scate bam
FILES=fragment_files/*
folder="scate_files/progen" #"scate_files/B_cells" #"mono"
mkdir -p $folder/bed_files
# create bed files out of fragment files where cell barcode matches the barcodes of interest
for f in $FILES
do
  filename=$(basename $f)
  filename=${filename#*_} #removes everything before first occurence of _
  filename=${filename%_*} #removes everything after last occurence of _
	awk -F '\t' 'NR==FNR{a[$7];next} $4 in a {print $1"\t"$2"\t"$3"\t"NR "#" "'$filename'" }'  $folder/barcodes $f  > $folder/bed_files/"$filename.bed"
	echo "$f done!"
done
#bed to bam conversion
FILES=${folder}/bed_files/*
mkdir -p $folder/bam_files
for f in $FILES
do
  filename=$(basename $f)
  filename=${filename%.*}
  tr ' ' '\t' < $f 1<> $f # to convert single space to tab
  bedToBam -i $f -g /home/eramamur/resources/genomes/hg19/hg19.chrom.sizes > $folder/bam_files/"$filename.bam"
done
#to remove all bed files with 0 lines
find ${folder}/bed_files/ -type f -exec awk -v x=1 'NR==x{exit 1}' {} \; -exec rm -f {} \;
# to remove corresponding bam FILES
find ${folder}/bam_files/ -type 'f' -size -126c -delete
