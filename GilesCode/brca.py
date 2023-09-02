#!/usr/bin/env python
# coding: utf-8

# # Load data, train model

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from helper import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *
import matplotlib.pyplot as plt
from scipy import stats

from os.path import join
import warnings 

warnings.filterwarnings('ignore')



# In[2]:


# Load data
data = pd.read_csv('../Data/brca_small.csv')
X = data.values[:, :-1]
Y = data.values[:, -1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=100, random_state=1)

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# In[3]:


def fit_model(x, y, x_val, y_val):
    # Cross validate for C
    C_list = np.arange(0.1, 1.0, 0.05)
    best_loss = np.inf
    best_C = None

    for C in C_list:
        # Fit model
        model = LogisticRegression(C=C, penalty='l1', multi_class='multinomial',
                                   solver='saga', max_iter=20000)
        model.fit(x, y)

        # Calculate loss
        val_loss = log_loss(y_val, model.predict_proba(x_val))

        # See if best
        if val_loss < best_loss:
            best_loss = val_loss
            best_C = C
            
    # Train model with all data
    model = LogisticRegression(C=best_C, penalty='l1', multi_class='multinomial',
                               solver='saga', max_iter=10000)
    model.fit(np.concatenate((x, x_val), axis=0),
              np.concatenate((y, y_val), axis=0))
    
    return model


# In[4]:


# Train model
model = fit_model(X_train, Y_train, X_val, Y_val)


#%%
# OK, now let's get a gradient and hessian

d = X_train.shape[1]

BETA = model.coef_
A = model.intercept_.reshape(4,1)


def modelf(x):
    yhat = np.exp(A+np.dot(BETA,x.T))
    #return yhat.item() if x.shape[0]==1 else yhat
    return yhat[1]/np.sum(yhat)


def modelg(x):
    yhat = np.exp(A+np.dot(BETA,x.T))
    yhat = yhat/np.sum(yhat)
    
    return BETA[1]*yhat[1] - yhat[1]*np.dot(yhat.T,BETA)
    
def modelH(x):
    yhat = np.exp(A+np.dot(BETA,x.T))
    yhat = yhat/np.sum(yhat)
    
    return yhat[1]*(np.outer(BETA[1],BETA[1].T) -
        np.dot(np.dot(BETA.T,np.diagflat(yhat)),BETA) +
        np.outer(np.dot(yhat.T,BETA),np.dot(yhat.T,BETA)))




xloc = X_train[1].reshape(1,100)

modelf(xloc)

gradient = modelg(xloc)
hessian = modelf(xloc)

#%%

# Now to set up D matrices

feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)

u, s, vh = np.linalg.svd(cov_mat, full_matrices=True)
K = 10000
s_max = s[0]
min_acceptable = s_max/K
s2 = np.copy(s)
s2[s <= min_acceptable] = min_acceptable
cov2 = np.matmul(u, np.matmul(np.diag(s2), vh))

M_linear = 1000 # 10 seconds/1000 perms or so
D_matrices = make_all_lundberg_matrices(M_linear, cov2)


#%%

# Simulation


n_pts = 20
nsim_per_point = 50
M = 1000
n_samples_per_perm = 10
K = 50
n_boot = 100

np.random.seed(1)
X_locs = X_test



kshaps_indep = []
kshaps_dep = []
sss_indep = []
sss_dep = []

for i in range(n_pts):
    xloci = X_test[i].reshape((1,d))
    gradient = modelg(xloci).T
    hessian = modelH(xloci)
    
    shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov2)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient)
    
        
    sims_kshap_indep = []
    sims_kshap_dep = []
    sims_ss_indep = []
    sims_ss_dep = []
    for j in range(nsim_per_point):
        print([i,j])
        
        independent_features=True
        try: 
            obj_kshap_indep = cv_kshap_compare(modelf, X_train, xloci,
                                independent_features,
                                gradient, hessian,
                                shap_CV_true=shap_CV_true_indep,
                                M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot) 
        except:
            print('kshap indep exception')
            obj_kshap_indep = []
            for _ in range(8):
                obj_kshap_indep.append( np.repeat(float('nan'),len(shap_CV_true_indep)))
            
        sims_kshap_indep.append(obj_kshap_indep)
            
        try:
            obj_ss_indep = cv_shapley_sampling(modelf, X_train, xloci, 
                                    independent_features,
                                    gradient, hessian, shap_CV_true=shap_CV_true_indep,
                                    M=np.floor_divide(M,10), n_samples_per_perm=n_samples_per_perm)
        except:
            print('ss indep exception')
            obj_ss_indep = []
            for _ in range(4):
                obj_ss_indep.append( np.repeat(float('nan'),len(shap_CV_true_indep)))
                        
        sims_ss_indep.append(obj_ss_indep)
            
        independent_features=False
        try:
            obj_kshap_dep = cv_kshap_compare(modelf, X_train, xloci,
                                independent_features,
                                gradient,
                                shap_CV_true=shap_CV_true_dep,
                                M=M, n_samples_per_perm=n_samples_per_perm, cov_mat=cov2, 
                                K = K, n_boot=n_boot)
        except:
            print('kshap dep exception')
            obj_kshap_dep = []
            for _ in range(8):
                obj_kshap_dep.append( np.repeat(float('nan'),len(shap_CV_true_dep)))
                
        sims_kshap_dep.append(obj_kshap_dep)
        
        try:
            obj_ss_dep = cv_shapley_sampling(modelf, X_train, xloci, 
                                    independent_features,
                                    gradient, shap_CV_true=shap_CV_true_dep,
                                    M=np.floor_divide(M,10), n_samples_per_perm=n_samples_per_perm, cov_mat=cov2)
        except:
            print('ss dep exception')
            obj_ss_dep = []
            for _ in range(4):
                obj_ss_dep.append( np.repeat(float('nan'),len(shap_CV_true_indep)))
                
        sims_ss_dep.append(obj_ss_dep)
       


    kshaps_indep.append(sims_kshap_indep)
    kshaps_dep.append(sims_kshap_dep)
    sss_indep.append(sims_ss_indep)
    sss_dep.append(sims_ss_dep)

    np.save('brca_kshap_indep.npy',kshaps_indep)
    np.save('brca_kshap_dep.npy',kshaps_dep)
    np.save('brca_ss_indep.npy',sss_indep)
    np.save('brca_ss_dep.npy',sss_dep)


