# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:14:45 2023

@author: giles
"""


import numpy as np
import statsmodels.regression.linear_model as lm
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


# ## Generate data
# ### Data will have dependency, but we'll run both methods on it

#%%


d = 10
FEATURE_MEANS = np.repeat(0, d)
FEATURE_VARS = np.repeat(1, d)
FEATURE_COVS = [0.5, 0.25]
COV_MAT = make_banded_cov(FEATURE_VARS, FEATURE_COVS)


# Randomly generate samples
np.random.seed(15)
X = np.random.multivariate_normal(FEATURE_MEANS, COV_MAT, size=1000)
feature_means = np.mean(X, axis=0)
cov_mat = np.cov(X, rowvar=False)
xloc = np.random.multivariate_normal(FEATURE_MEANS, COV_MAT, size=1)


#%%
# ## Define Logistic Regression model
# - All we need is the $d$ coefficients
# - We don't need to actually fit the model; pretend it represents some $\{(X_i,Y_i)\}_{i=1}^n$




np.random.seed(1)
BETA = np.random.normal(0, 1, size = d)
def model(x):
    yhat = sigmoid(np.dot(x, BETA))
    #return yhat.item() if x.shape[0]==1 else yhat
    return yhat

#%%

# Calculate D matrices


independent_features=False
D_matrices = make_all_lundberg_matrices(10000, cov_mat) # Takes a while

#%%

n_pts = 40
nsim_per_point = 50
M = 1000
n_samples_per_perm = 10
K = 50
n_boot = 100

simstr = 'npt'+str(n_pts)+'nsim'+str(nsim_per_point)+'M'+str(M)+'npp'+str(n_samples_per_perm)+'K'+str(K)+'nboot'+str(n_boot)

np.random.seed(1)
X_locs = np.random.multivariate_normal(FEATURE_MEANS, COV_MAT, size=n_pts)

np.save('newlogistic_X.npy',X)
np.save('newlogistic_xloci.npy',X_locs)
np.save('newlogistic_beta.npy',BETA)


kshaps_indep = []
kshaps_dep = []
sss_indep = []
sss_dep = []

for i in range(n_pts):
    xloci = X_locs[i].reshape((1,d))
    gradient = logreg_gradient(model, xloci, BETA)
    hessian = logreg_hessian(model, xloci, BETA)
    
    shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient)
    
        
    sims_kshap_indep = []
    sims_kshap_dep = []
    sims_ss_indep = []
    sims_ss_dep = []
    for j in range(nsim_per_point):
        print([i,j])
        

        independent_features=True
        obj_kshap_indep = cv_kshap_compare(model, X, xloci,
                            independent_features,
                            gradient, hessian,
                            shap_CV_true=shap_CV_true_indep,
                            M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot)        
        sims_kshap_indep.append(obj_kshap_indep)
        
        obj_ss_indep = cv_shapley_sampling(model, X, xloci, 
                                independent_features,
                                gradient, hessian, shap_CV_true=shap_CV_true_indep,
                                M=M, n_samples_per_perm=n_samples_per_perm)
        sims_ss_indep.append(obj_ss_indep)
        
        independent_features=False
        obj_kshap_dep = cv_kshap_compare(model, X, xloci,
                            independent_features,
                            gradient,
                            shap_CV_true=shap_CV_true_dep,
                            M=M, n_samples_per_perm=n_samples_per_perm, cov_mat=COV_MAT, 
                            K = K, n_boot=n_boot)
        sims_kshap_dep.append(obj_kshap_dep)
        
        obj_ss_dep = cv_shapley_sampling(model, X, xloci, 
                                independent_features,
                                gradient, shap_CV_true=shap_CV_true_dep,
                                M=M, n_samples_per_perm=n_samples_per_perm, cov_mat=COV_MAT)
        sims_ss_dep.append(obj_ss_dep)

    kshaps_indep.append(sims_kshap_indep)
    kshaps_dep.append(sims_kshap_dep)
    sss_indep.append(sims_ss_indep)
    sss_dep.append(sims_ss_dep)

    np.save('newlogistic_kshap_indep.npy',kshaps_indep)
    np.save('newlogistic_kshap_dep.npy',kshaps_dep)
    np.save('newlogistic_ss_indep.npy',sss_indep)
    np.save('newlogistic_ss_dep.npy',sss_dep)
