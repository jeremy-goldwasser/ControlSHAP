#%%
import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('../FullSim2')
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

from sklearn.model_selection import train_test_split
#%%


n_pts = 40
nsim_per_point = 50
M=1000
n_samples_per_perm=10
K=50
n_boot=100

fname = 'credit_rf'

X_train = np.load('../Data/credit/X_train.npy')
y_train = np.load('../Data/credit/y_train.npy')

X_test = np.load('../Data/credit/X_test.npy')
y_test = np.load('../Data/credit/y_test.npy')
print(X_test.shape)
mapping_dict = pickle.load(open('../Data/credit/creditmapping.p','rb'))
# %%

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
    min_acceptable = s_max/Kr
    s2 = np.copy(s)
    s2[s <= min_acceptable] = min_acceptable
    cov2 = np.matmul(u, np.matmul(np.diag(s2), vh))
    
cov_mat = (cov2 + cov2.T)/2

#%%
rf = RandomForestClassifier().fit(X_train, y_train)
print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
print("Estimation accuracy: {}".format(np.mean((rf.predict(X_test) > 0.5)==y_test)*100))


def model(xloc):
    return rf.predict_proba(xloc)[:,1]
# %%
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

# %%
X_locs = X_test[np.random.choice(X_test.shape[0],n_pts,replace=False)]

# %%

# i = 0
for i in range(10):
    print(i)
    xloci = X_locs[i].reshape((1,d))
    gradient = gradfn(model,xloci,sds)
    # hessian = hessfn(model,xloci,sds)

    # shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat, mapping_dict)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient,mapping_dict=mapping_dict)

    independent_features=False
    obj_kshap_dep = cv_kshap_compare(model, X_train, xloci,
                        independent_features,
                        gradient,
                        shap_CV_true=shap_CV_true_dep,
                        M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot, 
                        cov_mat=cov_mat, 
                        mapping_dict=mapping_dict)
    print(obj_kshap_dep[0][0])
    
    # %%
for i in range(10):
    print(i)
    xloci = X_locs[i].reshape((1,d))
    gradient = gradfn(model,xloci,sds)
    # hessian = hessfn(model,xloci,sds)

    # shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat, mapping_dict)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient,mapping_dict=mapping_dict)

    independent_features=False
    obj_ss_dep = cv_shapley_sampling(model, X_train, xloci, 
                                        independent_features,
                                        gradient, shap_CV_true=shap_CV_true_dep,
                                        M=M, n_samples_per_perm=n_samples_per_perm, cov_mat=cov_mat,
                                        mapping_dict=mapping_dict)
    print(obj_ss_dep[0][0])


# %%
