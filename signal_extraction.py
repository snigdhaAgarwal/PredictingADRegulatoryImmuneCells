import re
from scipy.io import mmread
from itertools import *
import os
from ucscgenome import Genome
import numpy as np
from Bio import SeqIO
import gzip
import pyfasta
import glob
import csv
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import matplotlib.pyplot as plxt
from scipy.stats import norm, boxcox
from scipy.special import boxcox1p
import collections
import shutil
path = '/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/satpathy_blood_scatac/'

def saveClusterIndices():
    # get indices of cluster specific cells
    cluster_indices = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    open_file = open(path+'GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt')
    for i,line in enumerate(open_file):
        columns = line.split()
        cluster = columns[2]
        index = i - 1 # ignoring first line in barcodes.txt
        if cluster in ['Cluster'+str(a) for a in range(1,10)]:
            cluster_indices[1].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(10,12)]:
            cluster_indices[2].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(12,14)]:
            cluster_indices[3].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(14,17)]:
            cluster_indices[4].append(index)
        elif cluster == 'Cluster17':
            cluster_indices[5].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(18,21)]:
            cluster_indices[6].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(21,26)]:
            cluster_indices[7].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(26,32)]:
            cluster_indices[8].append(index)
    print("Number of cells in each cluster:")
    for key in cluster_indices:
        print(key, len(cluster_indices[key]))
    with open('eggs.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        for key, values in cluster_indices.items():
            spamwriter.writerow([key] + values)

def generateSignals():

    # load sparse matrix
    '''TODO: For each cluster, go over all peaks in peaks.txt.
    The label file for each cluster will contain all peaks.'''
    peak_by_cell_matrix = mmread(path+'GSE129785_scATAC-Hematopoiesis-All.mtx')
    matrix_csr = peak_by_cell_matrix.tocsr()


    cluster_indices = {}
    with open('eggs.csv', 'r') as f:
        for line in f:
            items = line.split()
            key, values = items[0], items[1:]
            cluster_indices[int(key)] = list(map(int, values))
    print("Done with cluster indices")

    # go over each peak in every bed files
    total_reads = 0 # To calculate RPM
    for cluster in range(1,9):
        columns = cluster_indices[cluster]
        #extract columns from matrix
        sum = matrix_csr[:,columns].sum(axis=1)
        total_reads = matrix_csr[:,columns].sum()
        print(cluster, total_reads)
        rpm = (sum*pow(10,6))/total_reads
        np.savetxt(str(cluster)+'label',rpm,fmt='%.2f')
    print("Done with label file generation")


def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

def generateOneHotEncodedSequences(peak_list, sequence_file, label_file):
    genome_dir = '/home/eramamur/resources/genomes/hg19'
    genomeObject = Genome('hg19', cache_dir=genome_dir, use_web=False)
    label_list = []
    final_sequences = []
    for (seq,labelseq) in peak_list:
        # Normal sequence encoding and appending
        chromosome = seq[0]
        #extracting chromosome number from chromosomes of form chr1_KI270710v1_random
        # label = chromosome.split('chr')[1].split('_')[0]
        label_list.append(labelseq)
        start = int(seq[1])
        end = int(seq[2])
        sequence = genomeObject[chromosome][start:end]
        encodedSequence = oneHotEncodeSequence(sequence)
        final_sequences.append(encodedSequence)
        '''Create fasta for each entry and then generate reverse
        complement and store back in peak_map'''
        # want more samples for higher valued peaks
        if labelseq > 1.0000:
            ofile = open("temp.fa", "w")
            ofile.write(">" + str(seq) + "\n" +sequence + "\n")
            ofile.close()
            records = [rec.reverse_complement(id="rc_"+rec.id, description = "reverse complement")
            for rec in SeqIO.parse("temp.fa", "fasta")]
            SeqIO.write(records, "temp.fa", "fasta")
            f = pyfasta.Fasta("temp.fa")
            for header in f.keys():
                sequence = str(f[header])
                encodedSequence = oneHotEncodeSequence(sequence)
                label_list.append(labelseq)
                final_sequences.append(encodedSequence)
    final_sequences = np.stack(final_sequences, axis=0)
    label_list = np.stack(label_list,axis=0)
    # return final_sequences,label_list
    np.save(sequence_file, final_sequences)
    np.save(label_file, label_list)
    # to remove the temp files created above
    for filename in glob.glob("temp*"):
        os.remove(filename)


def generateSingleLabelData(label_index):
    #Create training, test and validation data
    trainingSet=[]
    testSet=[]
    validationSet=[]
    cluster = label_index + 1
    #Plotting label
    labelL = []
    #creating and counting number of elements in each bin
    # bins from 0-10,10-20,20-30,30-40 and 40-120
    bin_dict = {0:[],1:[],2:[],3:[],4:[]}
    bin_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    with open(path+"GSE129785_scATAC-Hematopoiesis-All.peaks.txt") as f1, open('signal_data/'+str(cluster)+'label') as f2:
        for item, label in zip(islice(f1,1,None),f2):
            label = float(label)
            # if label > 10.0000:
            peak = item.strip().split('_')
            chromosome = peak[0]
            tuple = (chromosome, peak[1], int(peak[2])+1)
            if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
                testSet.append((tuple,label))
            elif chromosome.startswith('chr4'):
                bin_list[int(label/10)]+=1
                validationSet.append((tuple,label))
            else:
                #labelL.append(label)

                # if int(label/10) in bin_dict:
                #     bin_dict[int(label/10)].append((tuple,label))
                # else:
                #     bin_dict[4].append((tuple,label))
                trainingSet.append((tuple,label))
    # sns.distplot(bin_list ,  kde = False)
    # plt.savefig("output.png")
    print(bin_list)
    import random
    # to create pre training data
    # for item,list in bin_dict.items():
    #     trainingSet += random.sample(list,700)
    print('Started one hot encoding')
    print(len(trainingSet),len(validationSet),len(testSet))
    generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
    generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
    generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')

def plot_peak_distribution(label_index):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    count_dict = {}
    with open('signal_data/'+str(label_index+1)+'label') as f:
        for line in f:
            line = line.strip()
            line = int(line)
            if line in count_dict:
                count_dict[line] = count_dict[line] + 1
            else:
                count_dict[line] = 1
    od = collections.OrderedDict(sorted(count_dict.items()))
    # print(od.keys())
    # print(od.values())

    ax.bar(od.keys(),od.values())
    fig.savefig('bar_plot.png')

def extractCellBarcodes():
    path = '/projects/pfenninggroup/machineLearningForComputationalBiology/snigdha_stuff/'
    barcodes = []
    with open(path+"GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt") as f1:
        for line in f1:
            columns = line.split()
            cluster = columns[2]
            if cluster == 'Cluster10' or cluster=='Cluster11':
                barcodes.append(columns[6])

    print("Cell barcodes Done", len(barcodes))
    new_path = '/projects/pfenninggroup/machineLearningForComputationalBiology/snigdha_stuff/fragment_files/'
    # outputPath = '/projects/pfenninggroup/machineLearningForComputationalBiology/snigdha_stuff/frag_out/'
    for filename in os.listdir(new_path):
        with open(filename.split('.')[0]+'_edit.tsv', 'w') as f_outfile:
            f = gzip.open(new_path+filename, 'rb')
            for line in f:
                columns = line.split()
                barcode = columns[3]
                if barcode in barcodes:
                    f_outfile.writerow(line)
            f.close()
        f_outfile.close()
        print(filename," done!")

def newPeaksExtract():
    trainingSet=[]
    testSet=[]
    validationSet=[]
    with open('dendritic_centered.bed', 'w') as f_outfile:
        with open("dendritic_peaks.bed") as f1, open('dendritic_peaks_labels') as f2:
            for item, label in zip(f1,f2):
                label = float(label)
                peak = item.strip().split()
                chromosome = peak[0]
                tuple = (chromosome, peak[1], int(peak[2])+1)
                f_outfile.write('%s\t%d\t%d\n' % tuple)
                if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
                    testSet.append((tuple,label))
                elif chromosome.startswith('chr4'):
                    validationSet.append((tuple,label))
                else:
                    trainingSet.append((tuple,label))
    print('Started one hot encoding')
    print(len(trainingSet),len(validationSet),len(testSet))
    # generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
    # generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
    # generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')

def scatePeaks(bedFile):
    trainingSet=[]
    testSet=[]
    validationSet=[]
    with open(bedFile+'_centered.bed', 'w') as f_outfile:
        with open(bedFile+'.bed') as f1:
            for item in f1:
                peak = item.strip().split()
                chromosome = peak[0]
                # making all peaks equal size by centering around midpoint
                start = int(peak[1])
                end = int(peak[2])+1
                mid = (start + end)/2
                start = mid-250
                end = mid + 250
                tuple = (chromosome, start, end)
                f_outfile.write('%s\t%d\t%d\n' % tuple)
                label = np.log(float(peak[4]))
                if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
                    testSet.append((tuple,label))
                elif chromosome.startswith('chr4'):
                    validationSet.append((tuple,label))
                else:
                    trainingSet.append((tuple,label))
    print('Started one hot encoding')
    print(len(trainingSet),len(validationSet),len(testSet))
    generateOneHotEncodedSequences(trainingSet,bedFile+'_trainInput',bedFile+'_trainLabels')
    generateOneHotEncodedSequences(validationSet,bedFile+'_validationInput',bedFile+'_validationLabels')
    generateOneHotEncodedSequences(testSet,bedFile+'_testInput',bedFile+'_testLabels')

def scateCompare(file):
    l = list(map(list,zip(*[map(float,line.split()) for line in open(file)])))
    fig, ax2 = plt.subplots(ncols=1)
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax2.plot(l[0], l[1],'r.',alpha=0.1)
    # ax2.set_ylim(0,250)
    fig.tight_layout()
    fig.savefig("scateVsArchr.png")
    plt.close()

if __name__=="__main__":
    # saveClusterIndices()
    # generateSignals()

    # 1 = dendritic, 2 = monocytes, 3 = B cells
    label_index = 1

    # plot_peak_distribution(label_index)

    # generateTrainingData() # use later for multi label scenario

    # generateSingleLabelData(label_index)

    #To get all cell_barcodes related to cluster 10,11 - Dendritic
    # extractCellBarcodes()

    #To take a bed and a label file to generate one hot encodedSequence
    # newPeaksExtract()

    #extract scate peaks
    dendritic="scate.bed"
    mono="scate_files/B_cells/peaks"
    # for s in ['progen','basophil','cd4','cd8','dendritic','mono','nk_cells','B_cells']:
    #     scatePeaks("scate_files/"+s+"/peaks")
        # scatePeaks("scate_files/proper/"+s)
        # os.mkdir("scate_files/"+s+"/fdr_0.05/negatives")
        # for file in glob.glob("*.npy"):
        #     shutil.move(file,"scate_files/"+s+"/fdr_0.05/negatives/")

    # new archr peak generation
    scatePeaks("Archr/Cluster3")
    archrfile='out'
    originalFile = 'orig_out'
    # scateCompare(archrfile)
