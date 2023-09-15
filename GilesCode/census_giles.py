#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from helper2 import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper3_kshap import *
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# German Credit dataset has 1000 samples, 20 covariates; response is "good customer" (binary); ~70% of Ys are 1. On UCI ML Repo.
# 
# Problem: can't load as already categorical. The Gradient Boosting model they used allowed you to just input the (numerical) data w/ a list of the categorical indices -- we can't do that with sklrean logistic regression.
# - FWIW, I think we had to convert things manually for the bank dataset. I just don't want to have to deal with that again.
# - Actually, it might have been OK to begin with - I just made things more complicated than necessary. Wouldn't be the first time.

# In[3]:


# import sage 
# df = sage.datasets.credit()


# Census dataset has 30k samples, 12 covariates, binary response, ~75% of Ys are False; some features numerical, some categorical.
# 
# From UCI ML Repository. Predict whether income exceeds $50K/yr based on census data.

# In[4]:


X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)

print(X.shape)
X.head()
X_display.head() # No NAs
print(np.unique(y_display, return_counts=True))
X_binarized = pd.get_dummies(X_display)
print(X_binarized.shape)


# Make dictionary whose keys are all the original columns, & whose values are lists of all (could just be 1) the columns in the binarized dataset with those features.
# - This will be useful when we get around to fitting the SHAP values

# In[5]:


mapping_dict = {}
for i, col in enumerate(X_display.columns):
    bin_cols = []
    for j, bin_col in enumerate(X_binarized.columns):
        if bin_col.startswith(col):
            bin_cols.append(j)
    mapping_dict[i] = bin_cols
# mapping_dict
# print(np.sum(np.sum(X_train[:, 14:21], axis=1) != 1)) # Sanity check: It's right


# Rescale covariates to be between 0 and 1

# In[6]:


X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
y_int = y_display.astype("int8")

# Split into training & test sets


# In[7]:


np.random.seed(1)
n, d_bin = X_norm.shape
n_train = round(n*0.75)
train_idx = np.random.choice(n, size=n_train, replace=False)
# X_train, y_train = X_norm.iloc[train_idx], y_int[train_idx]
X_train_pd, y_train = X_norm.iloc[train_idx], y_int[train_idx]
X_train = X_train_pd.to_numpy()

test_idx = np.setdiff1d(list(range(n)), train_idx)
X_test_pd, y_test = X_norm.iloc[test_idx], y_int[test_idx]
X_test = X_test_pd.to_numpy()


# #### Train logistic regression model
# 
# 85% test accuracy, compared w/ 75% class imbalance. So model is pretty good.

# In[8]:


logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(round(np.mean(logreg.predict(X_test)==y_test)*100))


# ### Recreate logreg model in numpy
# ## NOTE: Previous iterations didn't have (or need) an intercept term

# In[9]:


BETA = logreg.coef_.reshape(d_bin)
INTERCEPT = logreg.intercept_

def model(x):
    yhat = sigmoid(np.dot(x, BETA) + INTERCEPT)
    return yhat.item() if x.shape[0]==1 else yhat

xloc = X_test[0:1]
print(logreg.predict_proba(X_test[0:1]))
print(model(xloc)) # Yes, our function matches sklearn
print(y_test[0]) # Correct classification


# ### Compute gradient & hessian wrt local x

# In[10]:


gradient = logreg_gradient(model, xloc, BETA)
hessian = logreg_hessian(model, xloc, BETA)


#%%
# Now lets try using lgb
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

def fmodel(xloc):
    return lgbmodel.predict(xloc)

sds = []
for i in range(d_bin):
    uu = np.unique(X_train[:,i])
    if len(uu) == 2:
        sds.append(uu)
    else:
        sds.append(np.repeat(np.std(X_train[:,i]),2))
sds = np.array(sds)



gradient = difference_gradient(fmodel,xloc,sds)

hessian = difference_hessian(fmodel,xloc,sds)

# # Compute SHAP values, assuming independent features
# #### Sanity check: Verify true SHAP values of the quadratic approximation add up to $f(x)-Ef(X)$

# In[11]:


feature_means = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)

avg_CV_empirical = np.mean(f_second_order_approx(fmodel(xloc),X_train, xloc, gradient, hessian))
pred = fmodel(xloc)
exp_CV_sum_empirical = pred - avg_CV_empirical
shap_CV_true_indep = compute_true_shap_cv_indep(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict=mapping_dict)
sum_shap_CV_true = np.sum(shap_CV_true_indep)
print(sum_shap_CV_true)
print(exp_CV_sum_empirical) # Yes, they're extremely close


# ## Shapley Sampling
# ### Bad/weird. Negative correlations for the important features; positive for unimportant.

# In[12]:


np.random.seed(13)
independent_features = True
obj_ss = cv_shapley_sampling(fmodel, X_train, xloc, 
                        independent_features,
                        gradient, hessian,
                        mapping_dict=mapping_dict,
                        M=100, n_samples_per_perm=10) # M is number of permutations
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_ss

order = np.argsort(np.abs(final_ests))[::-1]
print(final_ests[order]) # Final SHAP estimates, ordered
print(np.round(corr_ests[order], 2)) # Correlations
print(np.round(100*(corr_ests**2)[order])) # Variance reductions



# ## KernelSHAP
# #### Now things look good. 30-50%

# In[13]:


np.random.seed(1)
obj_kshap = cv_kshap(fmodel, X_train, xloc, 
            independent_features,
            gradient, hessian,
            mapping_dict=mapping_dict,
            M=1000, n_samples_per_perm=10, 
            var_method='boot', n_boot=250)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap
order = np.argsort(np.abs(final_ests))[::-1]
print(np.round(final_ests[order], 3)) # Final SHAP estimates, ordered
print(np.round(corr_ests[order], 2)) # Correlations
print(np.round(100*(corr_ests**2)[order])) # Variance reductions


# # Dependent Features
# ### Recondition covariance

# In[19]:


u, s, vh = np.linalg.svd(cov_mat, full_matrices=True)
print(s[0]/s[d_bin-1]) # Conditioning number is essentially infinite
s_max = s[0]
K = 10000 # Desired conditioning number
min_acceptable = s_max/K
s2 = np.copy(s)
s2[s <= min_acceptable] = min_acceptable
cov2 = np.matmul(u, np.matmul(np.diag(s2), vh))
# Basically identical
# print(cov2[:3,:3])
# print(cov_mat[:3,:3])


# ### Generate matrices to estimate true SHAP values
# - Takes 1m 15s for 1000 perms. 91x91 matrix so takes a while.

# In[20]:


D_matrices = make_all_lundberg_matrices(1000, cov2)


# ## Shapley Sampling
# Variance reductions around mid-90s!

# In[25]:


np.random.seed(1)
independent_features = False
shap_CV_true_dep = linear_shap_vals(xloc, D_matrices, feature_means, gradient)
obj_dep = cv_shapley_sampling(model, X_train, xloc,
                    independent_features,
                    gradient,
                    shap_CV_true=shap_CV_true_dep, # Equivalently, can give D_matrices instead
                    M=100,n_samples_per_perm=10,
                    mapping_dict=mapping_dict,
                    cov_mat=cov2)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_dep
order = np.argsort(np.abs(final_ests))[::-1]
print(final_ests[order]) # Final SHAP estimates, ordered
print(np.round(corr_ests[order], 2)) # Correlations
print(np.round(100*(corr_ests**2)[order])) # Variance reductions



# ## KernelSHAP
# Again, around 90% variance reduction!

# In[31]:


# get_ipython().run_line_magic('run', 'helper_dep')
np.random.seed(1)
independent_features = False
shap_CV_true_dep = linear_shap_vals(xloc, D_matrices, feature_means, gradient)
obj_kshap_dep = cv_kshap(model, X_train, xloc,
                    independent_features,
                    gradient,
                    shap_CV_true=shap_CV_true_dep,
                    M=1000,n_samples_per_perm=10,
                    mapping_dict=mapping_dict,
                    cov_mat=cov2)

final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap_dep
order = np.argsort(np.abs(final_ests))[::-1]
print(final_ests[order]) # Final SHAP estimates, ordered
print(np.round(corr_ests[order], 2)) # Correlations
print(np.round(100*(corr_ests**2)[order])) # Variance reductions



# In[ ]:





# In[ ]:




