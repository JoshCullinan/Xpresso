# Tensorflow program to replicate Xpresso work.
# Josh Culliinan
# 07/02/2021

import tensorflow as tf
import sys, gzip, h5py, pickle, os
import numpy as np
import pandas as pd
from mimetypes import guess_type
from Bio import SeqIO
from functools import partial
from scipy import stats
from IPython.display import Image

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from hyperopt import hp, STATUS_OK
import os

# Comment out to use Nvidia GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Uncomment line to potentially use code with TF2 --- may be other errors.
# tf.keras.backend.set_image_data_format('channels_last')

# Where data can be found -- please extract data from google drive to the data directory in your project folder
datadir = 'data'

#Xpresso data - unedited
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
train_hld, train_promoters, train_y, geneName_train = trainfile['data'], trainfile['promoter'], trainfile['label'], trainfile['geneName']
validfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
valid_hld, valid_promoters, valid_y, geneName_valid = validfile['data'], validfile['promoter'], validfile['label'], validfile['geneName']
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
test_hld, test_promoters, test_y, geneName_test = testfile['data'], testfile['promoter'], testfile['label'], testfile['geneName']

# #Edited Xpresso data + Methylation -- uncomment to use.
# trainfile = h5py.File(os.path.join(datadir, 'Newtrain.h5'), 'r')
# train_hld, train_promoters, train_y, geneName_train = trainfile['data'], trainfile['promoter'], trainfile['label'], trainfile['geneName']
# validfile = h5py.File(os.path.join(datadir, 'Newvalid.h5'), 'r')
# valid_hld, valid_promoters, valid_y, geneName_valid = validfile['data'], validfile['promoter'], validfile['label'], validfile['geneName']
# testfile = h5py.File(os.path.join(datadir, 'Newtest.h5'), 'r')
# test_hld, test_promoters, test_y, geneName_test = testfile['data'], testfile['promoter'], testfile['label'], testfile['geneName']

# These parameters control which part of the promoters are used in the training. 
leftpos = 3000 #Start point of promoter - 3000 default
rightpos = 13500 #End point for promoter - 13500 default

# Create the subsequences of the promoters using the posistions.
train_promoters = train_promoters[:,leftpos:rightpos,:]
test_promoters = test_promoters[:,leftpos:rightpos,:]
valid_promoters = valid_promoters[:,leftpos:rightpos,:]

# Hyperparameters for the CNN -- current is as the paper authors had them
params = {'batchsize' : 128, 'leftpos' : 3000, 'rightpos' : 13500, 'activationFxn' : 'relu', 'numFiltersConv1' : 2**7, 'filterLenConv1' : 6, 'dilRate1' : 1,
            'maxPool1' : 30, 'numconvlayers' : { 'numFiltersConv2' : 2**5, 'filterLenConv2' : 9, 'dilRate2' : 1, 'maxPool2' : 10, 'numconvlayers1' : { 'numconvlayers2' : 'two' } },
            'dense1' : 2**6, 'dropout1' : 0.00099, 'numdenselayers' : { 'layers' : 'two', 'dense2' : 2, 'dropout2' : 0.01546 } }


# Defining the CNN
mse = 1
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
activationFxn = params['activationFxn']
halflifedata = Input(shape=(train_hld.shape[1:]), name='halflife')
input_promoter = Input(shape=train_promoters.shape[1:], name='promoter')

x = Conv1D(int(params['numFiltersConv1']), int(params['filterLenConv1']), dilation_rate=int(params['dilRate1']), padding='same', input_shape=train_promoters.shape[1:],activation=activationFxn)(input_promoter)
x.shape
x = MaxPooling1D(int(params['maxPool1']))(x)

# Used to add convolutional layers. 
if params['numconvlayers']['numconvlayers1'] != 'one':
        maxPool2 = int(params['numconvlayers']['maxPool2'])
        x = Conv1D(int(params['numconvlayers']['numFiltersConv2']), int(params['numconvlayers']['filterLenConv2']), dilation_rate=int(params['numconvlayers']['dilRate2']), padding='same',activation=activationFxn)(x) #[2, 3, 4, 5, 6, 7, 8, 9, 10]
        x = MaxPooling1D(maxPool2)(x)
        if params['numconvlayers']['numconvlayers1']['numconvlayers2'] != 'two':
            maxPool3 = int(params['numconvlayers']['numconvlayers1']['maxPool3'])
            x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numFiltersConv3']), int(params['numconvlayers']['numconvlayers1']['filterLenConv3']), dilation_rate=int(params['numconvlayers']['numconvlayers1']['dilRate3']), padding='same',activation=activationFxn)(x) #[2, 3, 4, 5]
            x = MaxPooling1D(maxPool3)(x)
            if params['numconvlayers']['numconvlayers1']['numconvlayers2']['numconvlayers3'] != 'three':
                maxPool4 = int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['maxPool4'])
                x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['numFiltersConv4']), int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['filterLenConv4']), dilation_rate=int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['dilRate4']), padding='same', kernel_initializer='glorot_normal',activation=activationFxn)(x) #[2, 3, 4, 5]
                x = MaxPooling1D(maxPool4)(x)

x = Flatten()(x)
x = Concatenate()([x, halflifedata])
x = Dense(int(params['dense1']))(x)
x = Activation(activationFxn)(x)
x = Dropout(params['dropout1'])(x)
if params['numdenselayers']['layers'] == 'two':
    x = Dense(int(params['numdenselayers']['dense2']))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params['numdenselayers']['dropout2'])(x)
main_output = Dense(1)(x)


# Compile model -- using stochastic gradient descent to optimise the loss. ADAM does not seem to perform better. Other optimisers may perform better?
model = Model(inputs=[input_promoter, halflifedata], outputs=[main_output])
model.compile(SGD(lr=0.0005, momentum=0.9),'mean_squared_error', metrics=['mean_squared_error'])

# Model architecture
print(model.summary())
# Creates image of model -- comment out if experiencing errors or install necessary graphviz & pydot
modelfile = os.path.join(datadir, 'plotted_model.png')
plot_model(model, show_shapes=True, show_layer_names=True, to_file=modelfile)

# Train model on training set and eval on 1K validation set
check_cb = ModelCheckpoint('bestparams.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop_cb = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min')

# Convert data to NP array to stop type error occuring. ## Comment out later to see if it works on your machine
train_promoters = np.array(train_promoters)
train_hld = np.array(train_hld)
train_y = np.array(train_y)
valid_promoters = np.array(valid_promoters)
valid_hld = np.array(valid_hld)
valid_y = np.array(valid_y)

# Train
result = model.fit( x = [train_promoters, train_hld],
                    y = train_y, 
                    batch_size=int(params['batchsize']), 
                    shuffle="batch", 
                    epochs=100, 
                    validation_data=([valid_promoters, valid_hld], valid_y),
                    callbacks=[earlystop_cb, check_cb],
                    #workers=6,
                    #use_multiprocessing=True
                )

# Get MSE
mse_history = result.history['val_mean_squared_error']
mse = min(mse_history)

# Evaluate performance on test set using best learned model
best_file = os.path.join('bestparams.h5')
model = load_model(best_file)
print('Loaded results from:', best_file)
predictions_test = model.predict([test_promoters, test_hld], batch_size=64).flatten()
slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, test_y)
print('Test R^2 = %.3f' % r_value**2)
df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, test_y)), columns=['Gene','Pred','Actual'])
print('Rows & Cols:', df.shape)
df.to_csv( 'predictions.txt', index=False, header=True, sep='\t')

## Plot the real values against the predicted -- strange graph is expected.
import matplotlib.pyplot as plt
x = np.linspace(-2,2,100)
y = x
fig = plt.figure(figsize=(10,8))
plt.scatter(predictions_test, test_y, label = 'Real vs Predicted')
plt.plot(x,y, color='red')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.ylim(-2, 2) 
plt.xlim(-2, 2)
plt.grid(True)
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')
