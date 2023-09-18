
import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

from helper2 import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *
from helper_sim import *
from os.path import join
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#%% Set some model parameters

n_pts = 40
nsim_per_point = 50
M=1000
n_samples_per_perm=10
K=50
n_boot=100

#%% Load data and calculate summaries

fname = 'credit_rf'

import sage
df = sage.datasets.credit()
# Property, other installment, housing, job, status of checking act, credit history, purpose, savings, employment since, marital status, old debtors
n = df.shape[0]
X_df = df.drop(["Good Customer"], axis=1)
y = df["Good Customer"]

categorical_columns = [
    'Checking Status', 'Credit History', 'Purpose', #'Credit Amount', # It's listed but has 923 unique values
    'Savings Account/Bonds', 'Employment Since', 'Personal Status',
    'Debtors/Guarantors', 'Property Type', 'Other Installment Plans',
    'Housing Ownership', 'Job', #'Telephone', 'Foreign Worker' # These are just binary
]
X_binarized = pd.get_dummies(X_df, columns=categorical_columns)
d_bin = X_binarized.shape[1]

mapping_dict = {}
for i, col in enumerate(X_df.columns):
    bin_cols = []
    for j, bin_col in enumerate(X_binarized.columns):
        if bin_col.startswith(col):
            bin_cols.append(j)
    mapping_dict[i] = bin_cols

np.random.seed(1)
X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
n_train = round(n*0.8)
train_idx = np.random.choice(n, n_train, replace=False)
X_train, y_train = X_norm.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy()
test_idx = np.setdiff1d(np.arange(n),train_idx)
X_test, y_test = X_norm.iloc[test_idx].to_numpy(), y.iloc[test_idx].to_numpy()
d = X_train.shape[1] # dimension of binarized data



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
    
cov_mat = (cov2 + cov2.T)/2


#%% Train model

rf = RandomForestClassifier().fit(X_train, y_train)
print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
print("Estimation accuracy: {}".format(np.mean((rf.predict(X_test) > 0.5)==y_test)*100))


def model(xloc):
    return rf.predict_proba(xloc)[:,1]


#%% Define gradients, hessians D_matrices

def gradfn(model,xloc,sds):
    return difference_gradient(model,xloc,sds)


def hessfn(model,xloc,sds):
    return difference_hessian(model,xloc,sds)

sds = []
for i in range(d):
    uu = np.unique(X_train[:,i])
    if len(uu) == 2:
        sds.append(uu)
    else:
        sds.append(np.repeat(np.std(X_train[:,i]),2))
sds = np.array(sds)


M_linear = 1000
D_matrices = make_all_lundberg_matrices(M_linear, cov2)



#%% Run simulation

X_locs = X_test[np.random.choice(X_test.shape[0],n_pts,replace=False)]

fullsim(fname,X_locs,X_train,model,gradfn,hessfn,D_matrices,mapping_dict=mapping_dict,sds=sds,
            nsim_per_point=nsim_per_point,M=M,n_samples_per_perm=n_samples_per_perm,K=K, n_boot=n_boot)