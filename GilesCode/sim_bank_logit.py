


import numpy as np
from helper import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from os.path import join
# import warnings
# warnings.filterwarnings('ignore')


# # Load and preprocess data
# ## Load Data
# ### Original Dataset
# - In "sage" package (also from Su-in Lee's group)
# - Has categorical variables with multiple levels

# In[17]:


dirpath = "../Data/bank"
# dirpath = /PATH/TO/DATA
df_orig = pd.read_csv(join(dirpath, "df_orig.csv"))
df_orig # 17 columns, so 16 features


# ### Binarized Data
# - Preprocessed so multilevel categorical features get split into multiple columns - one per binary feature
# - Unsurprisingly, the conditioning number of this matrix is near-infinite; we'll need to address this.
# 

# In[15]:


X_train_raw = np.load(join(dirpath, "X_train.npy"))
X_test_raw = np.load(join(dirpath, "X_test.npy"))
Y_train = np.load(join(dirpath, "Y_train.npy"))
Y_test = np.load(join(dirpath, "Y_test.npy"))
full_dim = X_train_raw.shape[1] # dimension including all binarized categorical columns
X_df = pd.read_csv(join(dirpath, "X_df.csv"))

xloc_raw = X_test_raw[0].reshape((1,full_dim))

X_df # 48-dimensional


# ## Standardize data

# In[50]:


# make mean-zero unit-variance
trainmean, trainstd = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
def rescale(x, trainmean, trainstd):
    return (x - trainmean) / trainstd
X_train = rescale(X_train_raw, trainmean, trainstd)
X_test = rescale(X_test_raw, trainmean, trainstd)
xloc = rescale(xloc_raw, trainmean, trainstd)

feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)


# ## Prepare for dealing with multilevel columns

# In[7]:


df_orig.columns = df_orig.columns.str.replace(' ', '_')
categorical_cols = ['Job', 'Marital', 'Education', 'Default', 'Housing',
                    'Loan', 'Contact', 'Month', 'Prev_Outcome']
mapping_dict = get_mapping_dict(df_orig, X_df, X_train_raw, categorical_cols)
mapping_dict


#%%  Let's look at logistic regression

logreg = LogisticRegression(max_iter=1000).fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)

print(round(np.mean(np.argmax(Y_pred,axis=1)==Y_test)*100))

print("AUC Score")
print(roc_auc_score(Y_test,Y_pred[:,1]))

print("Test log likelihood")
print(np.sum( Y_test*np.log(np.array(Y_pred[:,1]))))

#%%  Defining a model 

BETA = logreg.coef_.reshape(X_test.shape[1])
INTERCEPT = logreg.intercept_

def model(x):
    yhat = sigmoid(np.dot(x, BETA) + INTERCEPT)
    return yhat.item() if x.shape[0]==1 else yhat




#%%

# For dependent features


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

n_pts = 50
nsim_per_point = 20

np.random.seed(1)
X_locs = X_test[np.random.choice(X_test.shape[0],500,replace=False)]


kshaps_indep = []
kshaps_dep = []
sss_indep = []
sss_dep = []

for i in range(n_pts):
    xloci = X_locs[i].reshape((1,full_dim))
    gradient = logreg_gradient(model, xloci, BETA)
    hessian = logreg_hessian(model, xloci, BETA)
    
    shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat, mapping_dict)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient,mapping_dict=mapping_dict)
    
    sims_kshap_indep = []
    sims_kshap_dep = []
    sims_ss_indep = []
    sims_ss_dep = []
    for j in range(nsim_per_point):
        print([i,j])
        independent_features=True
        obj_kshap_indep = cv_kshap_compare(model, X_train, xloci,
                            independent_features,
                            gradient, hessian,
                            shap_CV_true=shap_CV_true_indep,
                            M=1000, n_samples_per_perm=10, K = 100, n_boot=250,
                            mapping_dict=mapping_dict)        
        sims_kshap_indep.append(obj_kshap_indep)
        
        obj_ss_indep = cv_shapley_sampling(model, X_train, xloci, 
                                independent_features,
                                gradient, hessian, shap_CV_true=shap_CV_true_indep,
                                M=1000, n_samples_per_perm=10,
                                mapping_dict=mapping_dict)
        sims_ss_indep.append(obj_ss_indep)
        
        independent_features=False
        obj_kshap_dep = cv_kshap_compare(model, X_train, xloci,
                            independent_features,
                            gradient,
                            shap_CV_true=shap_CV_true_dep,
                            M=1000, n_samples_per_perm=10, cov_mat=cov2, 
                            K = 100, n_boot=250,
                            mapping_dict=mapping_dict)
        sims_kshap_dep.append(obj_kshap_dep)
        
        obj_ss_dep = cv_shapley_sampling(model, X_train, xloci, 
                                independent_features,
                                gradient, shap_CV_true=shap_CV_true_dep,
                                M=1000, n_samples_per_perm=10, cov_mat=cov2,
                                mapping_dict=mapping_dict)
        sims_ss_dep.append(obj_ss_dep)

    kshaps_indep.append(sims_kshap_indep)
    kshaps_dep.append(sims_kshap_dep)
    sss_indep.append(sims_ss_indep)
    sss_dep.append(sims_ss_dep)

    np.save('bank_logit_kshap_indep.npy',kshaps_indep)
    np.save('bank_logit_kshap_dep.npy',kshaps_dep)
    np.save('bank_logit_ss_indep.npy',sss_indep)
    np.save('bank_logit_ss_dep.npy',sss_dep)