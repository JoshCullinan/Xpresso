import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
import random

# Device configuration -- for use with GPUs.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Current dir
dirname = os.path.dirname(__file__)

#Load in pre-processed data from npy arrays 
# ### USING THEIR DATA ################################################
# print("Currently using Xpresso data withOUT methylation data")
# promoters = np.load('data/Original_promoters.npy')
# halflifedata = np.load('data/Original_halflifedata.npy')
# labels = np.load('data/Original_labels.npy')
# geneNames = np.load('data/Original_geneNames.npy')
# FCLayer = 1128
# ######################################################################

##### Using Methylation DATA ####################################
print("Currently using Xpresso data WITH methylation data")
promoters = np.load('data/promoters.npy')
halflifedata = np.load('data/halflifedata.npy')
labels = np.load('data/labels.npy')
geneNames = np.load('data/geneNames.npy')
FCLayer = 1129
#########################################################

#Used to alter test/train/validate counts - currently set to mimic expresso
valid_count = 1000
test_count = 1000
train_count = len(promoters) - valid_count - test_count

i = 0
if train_count > 0:
    X_trainpromoter = promoters[i:i+train_count,:]
    X_trainhalflife = halflifedata[i:i+train_count,:]
    geneName_train = geneNames
    y_train = labels[i:i+train_count]
i += train_count
if valid_count > 0:
    X_validpromoter = promoters[i:i+valid_count,:]
    X_validhalflife = halflifedata[i:i+valid_count,:]
    geneName_valid = geneNames[i:i+valid_count]
    y_valid = labels[i:i+valid_count]
i += valid_count
if test_count > 0:
    X_testpromoter = promoters[i:i+test_count,:]
    X_testhalflife = halflifedata[i:i+test_count,:]
    geneName_test = geneNames[i:i+test_count]
    y_test = labels[i:i+test_count]

# Alter this to change the region of the promoter used in the CNN. 
# To Do -- implement Optuna or SKlearn optimizer to alter these hyperparameters.
leftpos = 3000 #Start point of promoter
rightpos = 13500 #End point for promoter

# Hyper-parameters 
num_epochs = 100
batch_size = 128
learning_rate = 0.00005
patience = 10 # Early stopping
PATH = './cnn.pth' # Where to save best model.

# Create the subsequences of the promoters using the posistions.
X_trainpromoterSubseq = X_trainpromoter[:,leftpos:rightpos,:]
X_testpromoterSubseq = X_testpromoter[:,leftpos:rightpos,:]
X_validpromoterSubseq = X_validpromoter[:,leftpos:rightpos,:]

# Swap axes to make the input suitable to the CNN -- TF1 uses Batch x Input x Channels. 
# TF2 and PyTorch use Batch x Channels x Input, therefore must be edited to make suitable
X_trainpromoterSubseq = np.swapaxes(X_trainpromoterSubseq, 2, 1)
X_testpromoterSubseq = np.swapaxes(X_testpromoterSubseq, 2, 1)
X_validpromoterSubseq = np.swapaxes(X_validpromoterSubseq, 2, 1)

#Convert Train/Test/Validation sets to tensors
train_hld = torch.from_numpy(X_trainhalflife)
train_promoters = torch.from_numpy(X_trainpromoterSubseq)
train_y = torch.from_numpy(y_train)
valid_hld = torch.from_numpy(X_validhalflife)
valid_promoters = torch.from_numpy(X_validpromoterSubseq)
valid_y = torch.from_numpy(y_valid)
test_hld = torch.from_numpy(X_testhalflife)
test_promoters = torch.from_numpy(X_testpromoterSubseq)
test_y = torch.from_numpy(y_test)

## Create train and test datasets, then set up loaders for batching.
train_dataset = torch.utils.data.TensorDataset(train_hld, train_promoters, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = torch.utils.data.TensorDataset(valid_hld, valid_promoters, valid_y)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# test_dataset = torch.utils.data.TensorDataset(test_hld, test_promoters, test_y) # Not needed
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#import pdb; pdb.set_trace() #---Debug trace
#CNN Class -- Architecure the same as Xpresso
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels= 128, kernel_size=6, dilation=1, padding = 2, bias= True)
        self.pool1 = nn.MaxPool1d(30, padding = 2)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 32, kernel_size = 9, dilation = 1, padding = 2, bias= True)
        self.pool2 = nn.MaxPool1d(10, padding=2)
        self.fc1 = nn.Linear(FCLayer, 64) #Would be 1128 when using their data, 1129 when using methylation data.
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 1)
    
    def forward(self, promo, halflifedata):
        x = F.relu(self.conv1(promo))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x) 
        x = x.view(-1, 1120)
        x = torch.cat((x, halflifedata), 1) 
        x = F.relu(self.fc1(x))
        Drop1 = nn.Dropout(p=0.00099) # Value from Xpresso network
        x = Drop1(x) 
        x = F.relu(self.fc2(x))
        Drop2 = nn.Dropout(p=0.01546) # Also from xpresso network
        x = Drop2(x)       
        x = self.fc3(x)                     
        return x

# All tensors and models must be on the same device -- send to GPU/CPU
model = ConvNet().to(device)

# Criterion is the loss metric. Can use MSE, MAE, etc.
criterion = nn.MSELoss()

# Optimizer is SGD to replicate Xpresso. Uncomment to use ADAM
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate); print(f'using ADAM to optimise, with learning rate: {learning_rate}')
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9); print(f'using SGD to optimise, with learning rate: {learning_rate}')

# Import my EarlyStopTool.py -- not robust. Edit with care.
from EarlyStopTool import early_stopping
es = early_stopping(patience = patience, verbose = True, path = PATH)

#Needed due to batch nature of training.
def Average(lst): 
    return sum(lst) / len(lst)

# Training loop, with validation step every epoch.
for epoch in range(num_epochs):
    #model.train(True)
    loss_mem = []
    for i, (hld, promo, ls) in enumerate(train_loader):
        #Send to GPU and convert to correct format.
        promo = promo.to(device, dtype=torch.float)
        ls = ls.to(device, dtype=torch.float)
        hld = hld.to(device, dtype=torch.float)

        optimizer.zero_grad()

        #Cast to float16 (Half), better for speed and lower memory usage.
        with autocast(enabled=True):
            # Forward pass
            outputs = model(promo, hld)
            outputs = torch.squeeze(outputs)
            #Calculate Loss
            loss = criterion(outputs, ls)
            loss_mem.append(loss)

        # Backward and optimize
        loss.backward()
        optimizer.step()

    vloss_mem = []
    for i, (v_hld, v_promo, v_ls) in enumerate(valid_loader):
        # Don't want validation to effect loss or learning. Therefore, no grad.
        with torch.no_grad():
            v_proms = valid_promoters.to(device, dtype = torch.float)
            v_hld = valid_hld.to(device, dtype = torch.float)
            v_y = valid_y.to(device, dtype = torch.float)
            with autocast(enabled=True):
                voutputs = model(v_proms, v_hld)
                voutputs = torch.squeeze(voutputs)
                vloss_mem.append(criterion(voutputs, v_y))

    #Early stopping
    es(Average(vloss_mem), model)

    print (f'Epoch [{epoch+1}/{num_epochs}] ----> Training Loss: {Average(loss_mem):.6f} & Validation Loss: {Average(vloss_mem):.6f}')

    if es.early_stop:
        print(f'\nEarly Stopping. Best Validation Loss: {es.best_validation_score():.4f}')
        break

print('\nFinished Training\n')


#### Evaluate the model on the test set ####
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error

# Load in best model from training. 
best_model = ConvNet().to(device)
best_model.load_state_dict(torch.load(PATH))
best_model.eval()
with torch.no_grad():
    test_hld = test_hld.to(device, dtype=torch.float)
    test_promoters = test_promoters.to(device, dtype=torch.float)
    test_y = test_y.to(device, dtype=torch.float)

    with autocast(enabled=True):
        test_outputs = best_model(test_promoters, test_hld)
        test_outputs = torch.squeeze(voutputs)
        tloss = criterion(voutputs, v_y)
    
    test_R2 = r2_score(test_y.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
print(f'MSE of the model on the test set: {tloss:.3f}. R^2 for test set: {test_R2:.3f}.')


## Plot the real against the predicted -- Odd plot, represents the terrible R^2. 
import matplotlib.pyplot as plt
x = np.linspace(-2,2,100)
y = x
fig = plt.figure(figsize=(10,8))
plt.scatter(test_outputs.cpu().detach().numpy(), test_y.cpu().detach().numpy(), label = 'Real vs Predicted')
plt.plot(x,y, color='red')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.ylim(-2, 2) # consistent scale
plt.xlim(-2, 2) # consistent scale
plt.grid(True)
plt.tight_layout()
plt.show()
# fig.savefig('loss_plot.png', bbox_inches='tight')
