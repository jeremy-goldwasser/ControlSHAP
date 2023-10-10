#%% Import moduldes

import numpy as np
import pandas as pd

import sys
sys.path.append('../../HelperFiles')
from helper import *
from helper_dep import *
from helper_indep import *
from helper_shapley_sampling import *
from helper_kshap import *
from helper_sim import *
from os.path import join

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


n_pts = 40
nsim_per_point = 50
M=1000
n_samples_per_perm=[10,1]  # 1st for Kernel Shap, 2nd for Shapley Sampling
K=50
n_boot=100





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

fullsim(fname,X_locs,X_train,model,gradfn,hessfn,cov_mat,D_matrices,mapping_dict=mapping_dict,sds=sds,
            nsim_per_point=nsim_per_point,M=M,n_samples_per_perm=n_samples_per_perm,K=K, n_boot=n_boot)