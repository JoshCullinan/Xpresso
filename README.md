# Xpresso

Contains code to train and evaluate CNNs to predict mRNA expresso data using promoter sequences, half life data, and methylation data. 
Based on https://github.com/vagarwal87/Xpresso 

# How to
## Anaconda Download
Please download and install anaconda from: https://www.anaconda.com/products/individual#Downloads

Recommended to install two seperate conda environments using the two requirements.txt files.

## Download Data From Google Drive
Download the data from https://drive.google.com/file/d/1KJrtRByO8C0sgWg4rmAu7pdpaoLBzGtm/view?usp=sharing and extract to your project directory such that Your_Project_Folder\data\TheExtractedFiles is true.

## Create New Training Data
To save on the download size, new data must be created locally.
Run the newData.py file in your project directory with either conda environment.

## Replicating using TF1.py
To replicate Xpresso's work run the TF1.py file using the TF1 anaconda environment, ensure that the program uses the original data by altering the commented out data loading sections.

## Replicating using PyTorch.py
To replicate Xpresso's work run the PyTorch.py file using the PyTorch anaconda environment, ensure that the program uses the original data by altering the commented out data loading sections.

## New training
Train a new CNN using PyTorch using the PyTorch conda environment, the PyTorch.py commenting out the original data loading
