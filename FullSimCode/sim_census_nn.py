
import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
import pickle

from helper2 import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *
from helper_sim import *
from os.path import join
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split

#%% Set some model parameters

n_pts = 40
nsim_per_point = 50
M=1000
n_samples_per_perm=10
K=50
n_boot=100

#%% Load data and calculate summaries

fname = 'census_nn'

X_train = np.load('../Data/census/X_train.npy')
y_train = np.load('../Data/census/y_train.npy')

X_test = np.load('../Data/census/X_test.npy')
y_test = np.load('../Data/census/y_test.npy')

mapping_dict = pickle.load(open('../Data/census/censusmapping.p','rb'))


#%% Calculate Summaries

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
d = X_train.shape[1]
mapping_dict = None


feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)

u, s, vh = np.linalg.svd(cov_mat, full_matrices=True)
Kr = 10000
if np.max(s)/np.min(s) < Kr:
    cov2 = cov_mat
else:
    s_max = s[0]
    min_acceptable = s_max/K
    s2 = np.copy(s)
    s2[s <= min_acceptable] = min_acceptable
    cov2 = np.matmul(u, np.matmul(np.diag(s2), vh))
    
cov_mat = cov2


#%% Train model
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

# Convert the input and label data to PyTorch tensors
inputs = torch.tensor(X_train, dtype=torch.float32)
labels = torch.tensor(y_train, dtype=torch.long)

# Compute the class weights
class_counts = torch.bincount(labels)
num_samples = len(labels)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[labels]

# Create a sampler with balanced weights
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

# Create a DataLoader with the sampler
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)


torch.manual_seed(0)

# Create an instance
net = TwoLayerNet(input_size=d, hidden_size=50, output_size=2)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()#weight=torch.tensor(weights)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#.01

# Iterate over the training data in batches
num_epochs = 5

# Train the network for the specified number of epochs
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Zero the gradients for this batch
        optimizer.zero_grad()

        # Compute the forward pass of the network
        outputs = net(inputs)

        # Compute the loss for this batch
        loss = criterion(outputs, labels)

        # Compute the gradients of the loss with respect to the parameters
        loss.backward()

        # Update the parameters using the optimizer
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")



test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs = net(test_tensor)

n_test = X_test.shape[0]
n_positive_preds = torch.sum(np.argmax(outputs, axis=1)==1).item()
print("{}/{} predictions are for positive class; really {}"
    .format(n_positive_preds,n_test, np.sum(y_test==1)))
Y_preds = torch.argmax(outputs, axis=1)
print("Balanced sampling. {}% accuracy".format(round(100*(np.sum(y_test==Y_preds.numpy())/n_test))))


def neural_net(x):
    output = net(x)[0,1] if x.shape[0]==1 else net(x)[:,1]
    return output

def model(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return neural_net(x).detach().numpy()

def gradfn(model,x,sds):
    xloc_torch = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
    y_pred = net(xloc_torch)[0,1]
    y_pred.backward()
    gradient = xloc_torch.grad.detach().numpy().reshape((d, 1))
    return gradient


def hessfn(model,x,sds):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    hessian = torch.autograd.functional.hessian(neural_net, x)
    hessian = hessian.reshape((d,d)).detach().numpy()
    return hessian

#%% D-matrices

sds = None

M_linear = 1000
D_matrices = make_all_lundberg_matrices(M_linear, cov2)



#%% Run simulation

X_locs = X_test[np.random.choice(X_test.shape[0],n_pts,replace=False)]

fullsim(fname,X_locs,X_train,model,gradfn,hessfn,D_matrices,mapping_dict=mapping_dict,sds=sds,
            nsim_per_point=nsim_per_point,M=M,n_samples_per_perm=n_samples_per_perm,K=K,n_boot=n_boot)