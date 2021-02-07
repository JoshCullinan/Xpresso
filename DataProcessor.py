#Data Processor
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)

def one_hot(seq):
    seq_len = len(seq.item(0))
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((len(seq),seq_len,4), dtype='bool')
    for i in range(len(seq)):
        thisseq = seq.item(i)
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    return seq_vec



def preprocess(data_file):
    table = pd.read_table(data_file, index_col=0)
    maskedIDs = pd.read_table(os.path.join(dirname, 'data\mask_histone_genes_mm10.txt'), header=None) #mask histone genes, chrY genes already filtered out
    maskedIDs2 = pd.read_table(os.path.join(dirname, 'data\mask_histone_genes.txt'), header=None) #mask histone genes, chrY genes already filtered out
    table = table[~table.index.isin(maskedIDs[0])] #remove rows corresponding to chrY or histone sequences
    table = table[~table.index.isin(maskedIDs2[0])] #remove rows corresponding to chrY or histone sequences
    table[table.columns[[0,2,3,4,5,9]]] = np.log10(table[table.columns[[0,2,3,4,5,9]]]+0.1)
    table = table.sample(table.shape[0], replace=False, random_state=1)
    table[table.columns[range(0,10)]] = preprocessing.scale(table[table.columns[range(0,10)]])
    print("\nPre-processed data...one-hot encoding...")
    promoters = one_hot(table['PROMOTER'].values)
    halflifedata = table[table.columns[range(1,10)]].values
    labels = table['EXPRESSION'].values
    geneNames = list(table.index)
    print("Processed data from %s" % data_file)
    return promoters, halflifedata, labels, geneNames