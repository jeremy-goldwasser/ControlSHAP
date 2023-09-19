
#%% Set some model parameters

name="brca"
mod = "gbm"

n_pts = 40
nsim_per_point = 50
M=1000
n_samples_per_perm=[10,1]  # 1st for Kernel Shap, 2nd for Shapley Sampling
K=500
n_boot=100


#%% Import moduldes

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



#%% Load data and calculate summaries

fname = name+"_"+mod

X_train, y_train, X_test, y_test, mapping_dict = loaddata(name)



#%% Calculate Summaries

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
d = X_train.shape[1]

feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)
    
cov_mat = correct_cov(cov_mat,Kr=10000)

M_linear = 1000
D_matrices = make_all_lundberg_matrices(M_linear, cov_mat)


#%% Train model

model, gradfn, hessfn, sds = fitmodgradhess(mod,X_train,y_train,X_test,y_test)


#%% Run simulation

X_locs = X_test[np.random.choice(X_test.shape[0],n_pts,replace=False)]

fullsim(fname,X_locs,X_train,model,gradfn,hessfn,D_matrices,mapping_dict=mapping_dict,sds=sds,
            nsim_per_point=nsim_per_point,M=M,n_samples_per_perm=n_samples_per_perm,K=K, n_boot=n_boot)