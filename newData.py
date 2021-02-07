#Data merge
import pandas as pd
import numpy as np
import h5py
import os

dirname = os.path.dirname(__file__)
pd.set_option('display.max_rows', 10)

#Load in DNA Methylation data
Meth = pd.read_csv(os.path.join(dirname,'data\data_methylation_hm450.txt'), sep ='\t')
Meth.drop(axis=1, labels = 'Entrez_Gene_Id', inplace=True)

#Load in Expression data
Expression = pd.read_csv(os.path.join(dirname,'data\data_RNA_Seq_v2_expression_median.txt'), sep ='\t')
Expression.drop(axis=1, labels = 'Entrez_Gene_Id', inplace=True)

# Load in Hugo 2 Ensemble IDs data
HGC = pd.read_csv(os.path.join(dirname, 'data/HGNC2Ensembl.txt'), sep = '\t')
HGC.rename(columns={'Approved Symbol':'Hugo_Symbol'}, inplace=True)

# Innerjoin Methylation data and expression data with HUGO symbol to get ENSEMBLE ID
Meth = HGC.merge(Meth, on='Hugo_Symbol')
Expression = HGC.merge(Expression, on = 'Hugo_Symbol')

#Drop unnessesary columns
Meth.drop(labels = ['Hugo_Symbol', 'Previous Symbols', 'Synonyms'], axis =1, inplace=True)
Expression.drop(labels = ['Hugo_Symbol', 'Previous Symbols', 'Synonyms'], axis =1, inplace = True)

#Currently dropping duplicates, unsure if actually correct. TO DO: Test difference.
Meth.drop_duplicates(subset=['Ensembl Gene ID'], inplace=True)
Expression.drop_duplicates(subset=['Ensembl Gene ID'], inplace=True)

###### Criminal Task 1 -- Creates median methylation and Expression across all cancer types!! #######
Meth['Median Meth'] = Meth.iloc[:,1:].median(axis=1)
Expression['Median Exp'] = Expression.iloc[:,1:].median(axis=1)


# Extracts median methylation and median mRNA expression for each gene type. Definitelly stupid. 
# But mimics Xpresso?
Data = Expression[['Ensembl Gene ID', 'Median Exp']].merge( Meth[['Ensembl Gene ID', 'Median Meth']], how='inner', on='Ensembl Gene ID')

# ### FOR REFERENCE TO SHOW THAT THEY JUST TOOK THE MEDIAN ### 
# # Look at their data processing files from Git to confirm.
# epigen = pd.read_csv(os.path.join(dirname,'data/57epigenomes.RPKM.pc'), sep = '\t', error_bad_lines=False) # The dataset
# epigen['Median Meth'] = epigen.iloc[:,1:].median(axis=1) # Making median column
# print(epigen[['gene_id','Median Meth' ]].head())
# # Their median dataset, same Ensemble IDs but just median values for the 58 cell types.
# med57 = pd.read_csv(os.path.join(dirname,'data/57epigenomes.median_expr.txt'), sep = '\t', header=None)
# print(med57.head())


##Put subset of expression and methylation data into their file with halflife info and promotor, 
# save and send downstream.
Roadmap = pd.read_csv(os.path.join(dirname, 'data/Roadmap_FantomAnnotations.InputData.pM10Kb.txt'), sep='\t')

Merged_data = Data.merge(Roadmap, how = 'inner', left_on='Ensembl Gene ID', right_on='ENSID')
Merged_data.drop(labels = ['Ensembl Gene ID', 'EXPRESSION'], axis =1, inplace = True)
Merged_data.rename(columns={'Median Exp':'EXPRESSION', 'Median Meth':'METHYLATION'}, inplace=True)
cols = Merged_data.columns.tolist()
cols[0] = cols[2]
cols[1] = 'EXPRESSION'
cols[2] = 'METHYLATION'
Merged_data = Merged_data[cols]
Merged_data.sort_values(by='ENSID', axis=0, inplace=True)
print('Demonstrating the new dataset\n\n', Merged_data.head())

#Created array in same format as theirs. 
if not os.path.isfile(os.path.join(dirname, 'data/MergedDataSet.csv')):
    Merged_data.to_csv(path_or_buf = 'data/MergedDataSet.csv', sep ='\t', index = False, header=True, index_label=False)

data_file = os.path.join(dirname, 'data/MergedDataSet.csv')
from DataProcessor import one_hot, preprocess 
promoters, halflifedata, labels, geneNames = preprocess(data_file)

#Get geneNames to work in H5 format
geneNames = pd.Series(geneNames)
geneNames = geneNames.astype(str, copy = True)
geneNames = geneNames.str.replace('\'','')
geneNames = geneNames.str.replace('b','')
geneNames = geneNames.to_numpy(dtype='S15', copy=True)



###Create Train, Test, Valid Files ###
compress_args = {'compression': 'gzip', 'compression_opts': 1}
overwrite = input("Do you want to overwrite current files (Y/N):\n")
h5 = input('Do you want H5 outputs (Y/N): \n')
npy = input('Do you want numpy files (Y/N): \n')


### Final statements to save data into data directory ###
if h5 == 'y' or h5 == 'Y':
    if (os.path.isfile(os.path.join(dirname, 'data/Newtrain.h5')) and (overwrite == 'y' or overwrite == 'Y')) or (not os.path.isfile(os.path.join(dirname, 'data/Newtrain.h5'))):

        trainfile = os.path.join(dirname, 'data/Newtrain.h5')
        validfile = os.path.join(dirname, 'data/Newvalid.h5')
        testfile = os.path.join(dirname, 'data/Newtest.h5')

        h5f_train = h5py.File(trainfile, 'w')
        h5f_valid = h5py.File(validfile, 'w')
        h5f_test = h5py.File(testfile, 'w')

        import math
        valid_count = 1000
        test_count = 1000
        train_count = len(geneNames) - valid_count - test_count

        i = 0
        if train_count > 0:
            h5f_train.create_dataset('data'    , data=halflifedata[i:i+train_count,:], **compress_args)
            h5f_train.create_dataset('promoter', data=promoters[i:i+train_count,:], **compress_args)
            h5f_train.create_dataset('label'   , data=labels[i:i+train_count], **compress_args)
            h5f_train.create_dataset('geneName' , data=geneNames, **compress_args)
            h5f_train.close()
        i += train_count
        if valid_count > 0:
            h5f_valid.create_dataset('data'    , data=halflifedata[i:i+valid_count,:], **compress_args)
            h5f_valid.create_dataset('promoter', data=promoters[i:i+valid_count,:], **compress_args)
            h5f_valid.create_dataset('label'   , data=labels[i:i+valid_count], **compress_args)
            h5f_valid.create_dataset('geneName' , data=geneNames[i:i+valid_count],  **compress_args)
            h5f_valid.close()
        i += valid_count
        if test_count > 0:
            h5f_test.create_dataset('data'    , data=halflifedata[i:i+test_count,:], **compress_args)
            h5f_test.create_dataset('promoter', data=promoters[i:i+test_count,:], **compress_args)
            h5f_test.create_dataset('label'   , data=labels[i:i+test_count], **compress_args)
            h5f_test.create_dataset('geneName' , data=geneNames[i:i+test_count], **compress_args)
            h5f_test.close()

if npy == 'y' or npy == 'Y':
    if (os.path.isfile(os.path.join(dirname, 'data/promoters.npy')) and (overwrite == 'y' or overwrite == 'Y')) or not os.path.isfile(os.path.join(dirname, 'data/promoters.npy')):
        np.save(os.path.join(dirname, 'data/promoters.npy'), promoters)
        np.save(os.path.join(dirname, 'data/halflifedata.npy'), halflifedata)
        np.save(os.path.join(dirname, 'data/labels.npy'), labels)
        np.save(os.path.join(dirname, 'data/geneNames.npy'), geneNames)