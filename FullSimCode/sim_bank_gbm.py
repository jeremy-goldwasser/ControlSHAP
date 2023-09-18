

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

fname = 'bank_gbm'

dirpath = "../Data/bank"
# dirpath = /PATH/TO/DATA
df_orig = pd.read_csv(join(dirpath, "df_orig.csv"))

X_train_raw = np.load(join(dirpath, "X_train.npy"))
X_test_raw = np.load(join(dirpath, "X_test.npy"))
y_train = np.load(join(dirpath, "Y_train.npy"))
y_test = np.load(join(dirpath, "Y_test.npy"))
full_dim = X_train_raw.shape[1] # dimension including all binarized categorical columns
X_df = pd.read_csv(join(dirpath, "X_df.csv"))


trainmean, trainstd = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
def rescale(x, trainmean, trainstd):
    return (x - trainmean) / trainstd
X_train = rescale(X_train_raw, trainmean, trainstd)
X_test = rescale(X_test_raw, trainmean, trainstd)

feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)


df_orig.columns = df_orig.columns.str.replace(' ', '_')
categorical_cols = ['Job', 'Marital', 'Education', 'Default', 'Housing',
                    'Loan', 'Contact', 'Month', 'Prev_Outcome']


mapping_dict = get_mapping_dict(df_orig, X_df, X_train_raw, categorical_cols)
mapping_dict

d = X_df.shape[1]





#%% Calculate Summaries -- should not need to touch

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
d = X_train.shape[1]


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


#%% Train model -- only between model classes

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 8,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}

lgbmodel = lgb.train(params, d_train, 10000, valid_sets=[d_test])
print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
print("Estimation accuracy: {}".format(np.mean((lgbmodel.predict(X_test) > 0.5)==y_test)*100))


def model(xloc):
    return lgbmodel.predict(xloc)


#%% Define gradients, hessians D_matrices -- depends on model class

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