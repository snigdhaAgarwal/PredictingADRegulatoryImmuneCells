import argparse
import logging
import time

import numpy as np
import pandas as pd
from scipy.special import logit

import py2bit
from keras.models import load_model


def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}    
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

def getSequence(allele,chrom,position,left,right,genome):
    alleleLength = len(allele)
    deductable = "right"
    for i in range(alleleLength-1):
        if deductable=="right":
            right-=1
            deductable="left"
        elif deductable=="left":
            left-=1
            deductable="right"
    
    left_sequence = genome.sequence(chrom, position-left, position)
    right_sequence = genome.sequence(chrom, position+alleleLength, position+alleleLength+right)
    
    sequence = left_sequence.lower()+allele.lower()+right_sequence.lower()
    return sequence


def onBatchEnd(model,
    curr_batch_ref_sequences,
    curr_batch_alt_sequences,
    is_classifier,
    curr_batch_is_overlapping
    ):

   
    curr_batch_ref_sequences = np.stack(curr_batch_ref_sequences, axis=0)
    curr_batch_alt_sequences = np.stack(curr_batch_alt_sequences, axis=0)
    curr_batch_ref_scores = model.predict(curr_batch_ref_sequences)
    curr_batch_alt_scores = model.predict(curr_batch_alt_sequences)


    if is_classifier:
        curr_batch_effect_scores = logit(curr_batch_ref_scores) - logit(curr_batch_alt_scores)
    else:
        curr_batch_effect_scores = curr_batch_ref_scores - curr_batch_alt_scores
 

    logging.info("{0} New batch processing".format(time.asctime()))
    logging.info("{0} Ref sequences shape is {1}".format(time.asctime(), curr_batch_ref_sequences.shape))
    logging.info("{0} Alt sequences shape is {1}".format(time.asctime(), curr_batch_alt_sequences.shape))      
    logging.info("{0} Effect scores shape is {1}".format(time.asctime(), curr_batch_effect_scores.shape))

    scores_index = 0
    for overlap_val in curr_batch_is_overlapping:
        if overlap_val:
            values_to_print = [curr_batch_effect_scores[scores_index][0],
                               curr_batch_ref_scores[scores_index][0],
                               curr_batch_alt_scores[scores_index][0]]
            print("\t".join([str(val) for val in values_to_print]))
            scores_index+=1
        else:
            print("\t".join(["nan","nan","nan"])) 

def propagateVariantsAndGetScores(snp_info_file,
                                    overlap_info_file,
                                    model_path,
                                    is_classifier,
                                    batch_size,
                                    genome_path,
                                    left,
                                    right
                                ):

    genome_object = py2bit.open(genome_path)
    model = load_model(model_path, compile=False)
    curr_batch_ref_sequences = []
    curr_batch_alt_sequences = []
    curr_batch_is_overlapping = []
    variants_processed_overall = 0
    variants_propagated_overall = 0
    batches_propagated_overall = 0

    with open(snp_info_file, 'r') as sf, open(overlap_info_file, 'r') as of:
        sf_header = sf.readline()
        for sf_line in sf:
            of_line = of.readline()
            of_data = of_line.strip().split("\t")
            sf_data = sf_line.strip().split("\t")
            chrom = of_data[0]
            snp_ref_start = int(of_data[1])
            snp_ref_end = int(of_data[2])
            ref = sf_data[4]
            alt = sf_data[5]
            is_overlapping = int(of_data[4])

            
            if is_overlapping:
                ref_sequence = getSequence(ref,chrom,snp_ref_start,left,right,genome_object)
                alt_sequence = getSequence(alt,chrom,snp_ref_start,left,right,genome_object)
                ref_sequence_encoded = oneHotEncodeSequence(ref_sequence)
                alt_sequence_encoded = oneHotEncodeSequence(alt_sequence)
                curr_batch_ref_sequences.append(ref_sequence_encoded)
                curr_batch_alt_sequences.append(alt_sequence_encoded)
                curr_batch_is_overlapping.append(True)
            else:
                curr_batch_is_overlapping.append(False)

            if len(curr_batch_ref_sequences)==batch_size:
                onBatchEnd(model,
                    curr_batch_ref_sequences,
                    curr_batch_alt_sequences,
                    is_classifier,
                    curr_batch_is_overlapping,
                    )
                variants_propagated_overall += batch_size
                batches_propagated_overall += 1


                curr_batch_ref_sequences = []
                curr_batch_alt_sequences = []
                curr_batch_is_overlapping = []
                logging.info("{0} Total variants propagated {1}".format(time.asctime(), variants_propagated_overall))
                logging.info("{0} Total batches propagated {1}".format(time.asctime(), batches_propagated_overall))
                logging.info("{0} Total variants processed {1}".format(time.asctime(), variants_processed_overall))
            variants_processed_overall += 1


    #processing for incomplete batches at the end
    if curr_batch_is_overlapping:
        if curr_batch_ref_sequences:
            assert(len(curr_batch_ref_sequences) < batch_size)
            onBatchEnd(model,
                curr_batch_ref_sequences,
                curr_batch_alt_sequences,
                is_classifier,
                curr_batch_is_overlapping
                )
            variants_propagated_overall += len(curr_batch_ref_sequences)
            batches_propagated_overall += 1

        #remaining NA values need to be printed if last batch is full of non-overlapping regions
        else:
            for val in curr_batch_is_overlapping:
                assert(not val)
                print("\t".join(["nan","nan","nan"])) 
        logging.info("{0} Total variants propagated {1}".format(time.asctime(), variants_propagated_overall))
        logging.info("{0} Total batches propagated {1}".format(time.asctime(), batches_propagated_overall))
        logging.info("{0} Total variants processed {1}".format(time.asctime(), variants_processed_overall))

    genome_object.close()

if __name__=="__main__":

    parser = argparse.ArgumentParser("Propagate variants through a given model")
    parser.add_argument("-s","--snp-info-file",
        help="text file containing SNP information",
        required=True)
    parser.add_argument("-o","--overlap-info-file",
        help="bed-like file containing OCR overlap information",
        required=True)
    parser.add_argument("-m","--model-path",
        help="keras model hdf5 file path",
        required=True)
    parser.add_argument("-c","--classifier",
        action='store_true',
        default=False,
        help="include if this is a classifier",
        required=False)
    parser.add_argument("-b","--batch-size",
        default=500,
        type=int,
        help="batch size for propagation",
        required=False)
    parser.add_argument("-g",
                        "--genome",
                        default="/home/eramamur/resources/genomes/hg38/hg38.2bit",
                        help="path to genome reference 2bit file",
                        required=False
                        )
    parser.add_argument("-left",
                        "--left",
                        type=int,
                        default=499,
                        required=False
                        )
    parser.add_argument("-right",
                        "--right",
                        type=int,
                        default=500,
                        required=False
                        )
    parser.add_argument("-l",
                        "--log-file",
                        default="tanzi_variants_construct_sequence_and_propagate.log",
                        help="path to log file",
                        required=False
                        )
    
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO) 
    genome_path = args.genome
    left = args.left
    right = args.right
    model_path = args.model_path
    is_classifier = args.classifier 
    batch_size = args.batch_size
    snp_info_file = args.snp_info_file
    overlap_info_file = args.overlap_info_file


    propagateVariantsAndGetScores(snp_info_file,
                                overlap_info_file,
                                model_path,
                                is_classifier,
                                batch_size,
                                genome_path,
                                left,
                                right
                                )
