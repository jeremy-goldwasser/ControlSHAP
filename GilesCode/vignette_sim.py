#!/usr/bin/env python
# coding: utf-8

#%%


import numpy as np
import statsmodels.regression.linear_model as lm
from helper import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper3_kshap import *
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
# ### Compute gradient & hessian of model w.r.t. local x


gradient = logreg_gradient(model, xloc, BETA)
hessian = logreg_hessian(model, xloc, BETA)


#%%
# # Assume Independent Features
# ### Compute SHAP values of the quadratic approximation; verify they add up to $f(x)-Ef(X)$


yloc = model(xloc)
avg_CV_empirical = np.mean(f_second_order_approx(yloc,X, xloc, gradient, hessian))
pred = model(xloc)
exp_CV_sum_empirical = pred - avg_CV_empirical
shap_CV_true_indep = compute_true_shap_cv_indep(xloc, gradient, hessian, feature_means, cov_mat)
sum_shap_CV_true = np.sum(shap_CV_true_indep)
print(sum_shap_CV_true)
print(exp_CV_sum_empirical) # Yes, they're extremely close
print(exp_CV_sum_empirical - sum_shap_CV_true)

#%%

# ## Shapley Sampling values, assuming independent features
# ### Specify number of permutations
# - Don't need to give it true SHAP values of control variate; it computes them inside the function as above.


np.random.seed(15)
independent_features = True
obj_ss = cv_shapley_sampling(model, X, xloc, 
                        independent_features,
                        gradient, hessian,
                        M=100, n_samples_per_perm=10) # M is number of permutations)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_ss
print(corr_ests) # Pretty solid correlations
print(np.round(100*(corr_ests**2))) # Variance reductions


#%%
# ### Run until convergence
# - A SHAP estimate converges when its standard deviation, normalized by its absolute value, is below the given threshold.
# - Features with lowest SHAP values take longest to converge 

np.random.seed(15)
get_ipython().run_line_magic('run', 'helper_shapley_sampling')
obj_ss_auto = cv_shapley_sampling(model, X, xloc, 
                        independent_features,
                        gradient, hessian,
                        shap_CV_true=None, # Unnecessary; function calls compute_true_shap_cv_indep() 
                        M=None, n_samples_per_perm=10,
                        t=0.1, n_intermediate=50, # t is cutoff threshold; n_intermediate is evaluation frequency
                        verbose=True) # prints frequency of convergence for each feature
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_ss_auto
print(corr_ests) # Similarly solid correlations
print(rank_shap(final_ests)) # SHAP values, highest to lowest


#%%
# ## KernelSHAP, Assuming Indepdent Features
# ### Specifying number of permutations
# - Estimates variance and covariance of vanilla SHAP estimates via 250 bootstrapped samples. This is the default for kernelSHAP.

np.random.seed(1)
obj_kshap = cv_kshap(model, X, xloc, 
            independent_features,
            gradient, hessian,
            M=1000, n_samples_per_perm=10, 
            var_method='wls', n_boot=250)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap
print(np.round(100*(corr_ests**2))) # Variance reductions
print(rank_shap(final_ests)) # Same ranking on important features


#%%
# ### Run until convergence
# - KernelSHAP converges when 
# $\displaystyle \frac{\max_j \sigma(\hat{\varphi}_j)}{\max_j \hat{\varphi}_j - \min_j \hat{\varphi}_j} < t$.
# 
#     - That is, when the highest standard deviation of a feature, normalized by the difference between highest and lowest SHAP estimates, is below the given threshold.

np.random.seed(15)
obj_kshap = cv_kshap(model, X, xloc, 
            independent_features,
            gradient, hessian,
            M=None,n_samples_per_perm=10,
            t=0.05, n_intermediate=50,
            verbose=True)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap
print(np.round(100*(corr_ests**2))) # Variance reductions
print(rank_shap(final_ests)) # Same ranking on important features


#%%
# ### Grouped covariance
# - For more info on grouping, see the document I compiled.

np.random.seed(15)
obj_kshap_grouped = cv_kshap(model, X, xloc, 
            independent_features,
            gradient, hessian,
            M=1000, n_samples_per_perm=10, 
            var_method='grouped', K=50)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap_grouped
print(np.round(100*(corr_ests**2))) # Variance reductions
print(rank_shap(final_ests)) # Same ranking on important features


#%%
# ### WLS covariance

np.random.seed(1)
obj_kshap_wls = cv_kshap(model, X, xloc, 
            independent_features,
            gradient, hessian,
            M=1000, n_samples_per_perm=10, 
            var_method='wls')
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap_wls
print(corr_ests)
print(rank_shap(final_ests))


#%%
# Let's do a comparison
np.random.seed(1)
obj_kshap = cv_kshap_compare(model, X, xloc, 
            independent_features,
            gradient, hessian,
            M=1000, n_samples_per_perm=10, 
            n_boot=250)

vshap_ests_model, vshap_ests_CV, final_ests_boot, corr_ests_boot, final_ests_grouped, corr_ests_grouped, final_ests_wls, corr_ests_wls = obj_kshap

print(np.array([vshap_ests_model,final_ests_boot,final_ests_grouped,final_ests_wls]).T)
print(np.round(100*np.array([corr_ests_boot,corr_ests_grouped,corr_ests_wls]).T))
print(np.array([rank_shap(vshap_ests_model),rank_shap(final_ests_boot),rank_shap(final_ests_grouped),rank_shap(final_ests_wls)]).T)


#%%
# And now we should also do a simulation:
    
np.random.seed(1863)
n_iter = 200
independent_features = True
objs_kshap_indep = []
objs_ss_indep = []
for _ in range(n_iter):
    obj_indep = cv_kshap_compare(modebl, X, xloc, 
                independent_features,
                gradient, hessian,
                M=1000, n_samples_per_perm=10, 
                n_boot=50, K=20)
    objs_kshap_indep.append(obj_indep)
    
    obj_ss = cv_shapley_sampling(model, X, xloc, 
                            independent_features,
                            gradient, hessian,
                            M=1000, n_samples_per_perm=10)
    objs_ss_indep.append(obj_ss)


show_var_reducs_kshap(objs_kshap_indep, ylim_zero=True, verbose=True,message='Kernel Shap: Independent Samples')
show_var_reducs(objs_ss_indep, ylim_zero=True, verbose=True,message='Shapley Sampling: Independent Samples')


#%%
# # Dependent Features methods
# ## Estimate matrices used to approximate true SHAP values of linear model
# - This should be computationally intensive, since we can reuse it for many local x
# - Slow, since each permutation entails a matrix inversion

independent_features=False
D_matrices = make_all_lundberg_matrices(10000, cov_mat) # Takes a while

#%%
# ### Compute almost-true SHAP values of linear model; verify their sum is close to $f(x)-Ef(X)$

avg_CV_empirical = np.mean(f_first_order_approx(yloc,X, xloc, gradient))
pred = model(xloc)
exp_CV_sum_empirical = pred - avg_CV_empirical

avg_model_empirical = np.mean(model(X))
exp_model_sum_empirical = pred - avg_model_empirical
shap_CV_true_dep = linear_shap_vals(xloc, D_matrices, feature_means, gradient)
sum_shap_CV_true = np.sum(shap_CV_true_dep)
print(exp_CV_sum_empirical)
print(sum_shap_CV_true)


#%%
# ## Shapley Sampling values, assuming dependent features
# - For the sake of brevity, I won't show auto-convergence again.

np.random.seed(1)
shap_CV_true_dep = linear_shap_vals(xloc, D_matrices, feature_means, gradient)
obj_dep = cv_shapley_sampling(model, X, xloc,
                    independent_features,
                    gradient, # Don't need to give hessian, since CV is linear model 
                    shap_CV_true=shap_CV_true_dep, # Equivalently, can give D_matrices instead
                    M=100,n_samples_per_perm=10)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_dep
print(corr_ests)
print(rank_shap(final_ests))


#%%
# ## KernelSHAP values, assuming dependent features
# - Recall default method for variance estimation is bootstrapping w/ 250 resamplings.
#
np.random.seed(1)
obj_kshap_dep = cv_kshap(model, X, xloc,
                    independent_features,
                    gradient,
                    shap_CV_true=shap_CV_true_dep,
                    M=1000,n_samples_per_perm=10,K=50,var_method='grouped',)
final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = obj_kshap_dep
print(corr_ests)
print(rank_shap(final_ests))

#%%
# # Empirical variance reductions
# - We also can do this for Shapley Sampling or independent features. I'll just show it once, though.
#     - Takes ~20 seconds to run
#     
# ### I present 2 ways of estimating the correlation
# 1. "Average correlation" refers to taking the mean correlation across all iterations, for each feature
#     - We can do this because each iteration estimates the correlation between vanilla SHAP estimates of the model & its approximation
# 2. "Correlation between observations" refers to taking the correlation between those vanilla SHAP estimates across all iterations
#     - Unsurprisingly, it's less stable and closer to the empirical variance reduction
# 

np.random.seed(1)
n_iter = 20
independent_features = False
objs_kshap_dep = []
objs_ss_dep = []
for k in range(n_iter):
    print(k)
    #obj_dep = cv_kshap_compare(model, X, xloc,
    #                    independent_features,
    #                    gradient,
    #                    shap_CV_true=shap_CV_true_dep,
    #                    M=1000, n_samples_per_perm=10, K = 100, 
    #                    n_boot=250)
    #objs_kshap_dep.append(obj_dep)
    
    obj_ss = cv_shapley_sampling(model, X, xloc, 
                            independent_features,
                            gradient, shap_CV_true=shap_CV_true_dep,
                            M=1000, n_samples_per_perm=10)
    objs_ss_dep.append(obj_ss)


#show_var_reducs_kshap(objs_kshap_dep, ylim_zero=True, verbose=True,message='Kernel Shap: Dependent Samples')
show_var_reducs(objs_ss_dep, ylim_zero=True, verbose=True,message='Shapley Sampling: Dependent Samples')

#%%
# Let's compare estimates of correlation

corr_ests_boot_dep = np.array([objs_kshap_dep[i][3] for i in range(n_iter)])
corr_ests_boot_indep = np.array([objs_kshap_indep[i][3] for i in range(n_iter)])

corr_ests_grouped_dep = np.array([objs_kshap_dep[i][5] for i in range(n_iter)])
corr_ests_grouped_indep = np.array([objs_kshap_indep[i][5] for i in range(n_iter)])

corr_ests_wls_dep = np.array([objs_kshap_dep[i][7] for i in range(n_iter)])
corr_ests_wls_indep = np.array([objs_kshap_indep[i][7] for i in range(n_iter)])



#%%
# And check sums -- these seem reasonable. 

sum_vanilla_indep = np.array([np.sum(objs_kshap_indep[i][0]) for i in range(n_iter)])
sum_vanilla_dep = np.array([np.sum(objs_kshap_dep[i][0]) for i in range(n_iter)])

sum_CV_indep = np.array([np.sum(objs_kshap_indep[i][1]) for i in range(n_iter)])
sum_CV_dep = np.array([np.sum(objs_kshap_dep[i][1]) for i in range(n_iter)])

sum_boot_indep = np.array([np.sum(objs_kshap_indep[i][2]) for i in range(n_iter)])
sum_boot_dep = np.array([np.sum(objs_kshap_dep[i][2]) for i in range(n_iter)])

sum_grouped_indep = np.array([np.sum(objs_kshap_indep[i][4]) for i in range(n_iter)])
sum_grouped_dep = np.array([np.sum(objs_kshap_dep[i][4]) for i in range(n_iter)])

sum_wls_indep = np.array([np.sum(objs_kshap_indep[i][6]) for i in range(n_iter)])
sum_wls_dep = np.array([np.sum(objs_kshap_dep[i][6]) for i in range(n_iter)])



#%%
# #### A full simulation -- we'll look at n_sim = 200 simulations for each of n_pts = 100 
# data points to check out the over-all pattern of variance reductions and accuracy

n_pts = 40
nsim_per_point = 50
M = 1000
n_samples_per_perm = 10
K = 50
n_boot = 100

simstr = 'npt'+str(n_pts)+'nsim'+str(nsim_per_point)+'M'+str(M)+'npp'+str(n_samples_per_perm)+'K'+str(K)+'nboot'+str(n_boot)

np.random.seed(1)
X_locs = np.random.multivariate_normal(FEATURE_MEANS, COV_MAT, size=n_pts)

np.save('X'+simstr+'.npy',X)
np.save('xloci'+simstr+'.npy',X_locs)
np.save('beta'+simstr+'.npy',BETA)


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
                            shap_CV_true=shap_CV_true_dep,
                            M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot)        
        sims_kshap_indep.append(obj_kshap_indep)
        
        obj_ss_indep = cv_shapley_sampling(model, X, xloci, 
                                independent_features,
                                gradient, hessian, shap_CV_true=shap_CV_true_dep,
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

    np.save('kshap_indep_'+simstr+'.npy',kshaps_indep)
    np.save('kshap_dep_'+simstr+'.npy',kshaps_dep)
    np.save('ss_indep_'+simstr+'.npy',sss_indep)
    np.save('ss_dep_'+simstr+'.npy',sss_dep)

#%%
# Now an analysis.  We first want empirical variance reductions

kshaps_dep = np.array(kshaps_dep)
kshaps_indep = np.array(kshaps_indep)

sss_dep = np.array(sss_dep)
sss_indep = np.array(sss_indep)

means_kshap_dep = np.mean(kshaps_dep,axis=1)
means_kshap_indep = np.mean(kshaps_indep,axis=1)
vars_kshap_dep = np.var(kshaps_dep,axis=1)
vars_kshap_indep = np.var(kshaps_indep,axis=1)

means_ss_dep = np.mean(sss_dep,axis=1)
means_ss_indep = np.mean(sss_indep,axis=1)
vars_ss_dep = np.var(sss_dep,axis=1)
vars_ss_indep = np.var(sss_indep,axis=1)

var_reducs_indep = 1-np.array([[vars_ss_indep[i][0]/vars_ss_indep[i][1] for i in range(n_pts)],
                    #[vars_kshap_indep[i][2]/vars_kshap_indep[i][0] for i in range(n_pts)],
                    [vars_kshap_indep[i][6]/vars_kshap_indep[i][0] for i in range(n_pts)]])

reducs_indep_25 = np.reshape(np.quantile(var_reducs_indep,0.25,axis=1).T,[2*d])
reducs_indep_50 = np.reshape(np.quantile(var_reducs_indep,0.50,axis=1).T,[2*d])
reducs_indep_75 = np.reshape(np.quantile(var_reducs_indep,0.75,axis=1).T,[2*d])

xpts = np.repeat( np.arange(0,d), 2) + np.tile(np.array([-0.1,0.1]),d)
plt.errorbar(xpts,reducs_indep_50,yerr=np.array([reducs_indep_50-reducs_indep_25,reducs_indep_75-reducs_indep_50]),fmt='o')
plt.ylim([-1,1])


var_reducs_dep = 1-np.array([[vars_ss_dep[i][0]/vars_ss_dep[i][1] for i in range(n_pts)],
                    #[vars_kshap_dep[i][2]/vars_kshap_dep[i][0] for i in range(n_pts)],
                    [vars_kshap_dep[i][6]/vars_kshap_dep[i][0] for i in range(n_pts)]])


reducs_dep_25 = np.reshape(np.quantile(var_reducs_dep,0.25,axis=1).T,[2*d])
reducs_dep_50 = np.reshape(np.quantile(var_reducs_dep,0.50,axis=1).T,[2*d])
reducs_dep_75 = np.reshape(np.quantile(var_reducs_dep,0.75,axis=1).T,[2*d])

plt.errorbar(xpts,reducs_dep_50,yerr=np.array([reducs_dep_50-reducs_dep_25,reducs_dep_75-reducs_dep_50]),fmt='o')
plt.ylim([0,1])





#%% Bias in estiamted correlation between CV and control variates


mean_covest_indep = np.array([[means_ss_indep[i][3] for i in range(n_pts)],
                              [means_kshap_indep[i][3] for i in range(n_pts)],
                              [means_kshap_indep[i][7] for i in range(n_pts)]])/M

                    
tmp_ss = np.moveaxis(sss_indep,range(4),(1,3,0,2))
tmp_kshap = np.moveaxis(kshaps_indep,range(4),(1,3,0,2))      
                       
emp_covest_indep = np.array([[[np.cov(tmp_ss[0][i][j],tmp_ss[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)],
                             [[np.cov(tmp_kshap[0][i][j],tmp_ss[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)],
                             [[np.cov(tmp_kshap[0][i][j],tmp_kshap[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)]])
                             
covest_bias_indep = (mean_covest_indep - emp_covest_indep)



covest_bias_indep_25 = np.reshape(np.quantile(covest_bias_indep,0.25,axis=1).T,[3*d])
covest_bias_indep_50 = np.reshape(np.quantile(covest_bias_indep,0.50,axis=1).T,[3*d])
covest_bias_indep_75 = np.reshape(np.quantile(covest_bias_indep,0.75,axis=1).T,[3*d])


                             
mean_covest_dep = np.array([[means_ss_dep[i][3] for i in range(n_pts)],
                              [means_kshap_dep[i][3] for i in range(n_pts)],
                              [means_kshap_dep[i][7] for i in range(n_pts)]])/M


tmp_ss = np.moveaxis(sss_dep,range(4),(1,3,0,2))
tmp_kshap = np.moveaxis(kshaps_dep,range(4),(1,3,0,2))      
                       
emp_covest_dep = np.array([[[np.cov(tmp_ss[0][i][j],tmp_ss[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)],
                             [[np.cov(tmp_kshap[0][i][j],tmp_ss[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)],
                             [[np.cov(tmp_kshap[0][i][j],tmp_kshap[1][i][j])[0,1] for j in range(d)] for i in range(n_pts)]])
                             
covest_bias_dep = (mean_covest_dep - emp_covest_dep)                        


covest_bias_dep_25 = np.reshape(np.quantile(covest_bias_dep,0.25,axis=1).T,[3*d])
covest_bias_dep_50 = np.reshape(np.quantile(covest_bias_dep,0.50,axis=1).T,[3*d])
covest_bias_dep_75 = np.reshape(np.quantile(covest_bias_dep,0.75,axis=1).T,[3*d])


plt.errorbar(xpts,covest_bias_indep_50,yerr=np.array([covest_bias_indep_50-covest_bias_indep_25,covest_bias_indep_75-covest_bias_indep_50]),fmt='o')
plt.errorbar(xpts,covest_bias_dep_50,yerr=np.array([covest_bias_dep_50-covest_bias_dep_25,covest_bias_dep_75-covest_bias_dep_50]),fmt='o')


#%% sd in estimated correlation between CV and control variates

sd_corest_indep = np.array([[np.sqrt(vars_ss_indep[i][3]) for i in range(n_pts)],
                          [np.sqrt(vars_kshap_indep[i][3]) for i in range(n_pts)],
                          [np.sqrt(vars_kshap_indep[i][7]) for i in range(n_pts)]])

corest_indep_25 = np.reshape(np.quantile(sd_corest_indep,0.25,axis=1).T,[3*d])
corest_indep_50 = np.reshape(np.quantile(sd_corest_indep,0.50,axis=1).T,[3*d])
corest_indep_75 = np.reshape(np.quantile(sd_corest_indep,0.75,axis=1).T,[3*d])


plt.errorbar(xpts,corest_indep_50,yerr=np.array([corest_indep_50-corest_indep_25,corest_indep_75-corest_indep_50]),fmt='o')


sd_corest_dep = np.array([[np.sqrt(vars_ss_dep[i][3]) for i in range(n_pts)],
                          [np.sqrt(vars_kshap_dep[i][3]) for i in range(n_pts)],
                          [np.sqrt(vars_kshap_dep[i][7]) for i in range(n_pts)]])

corest_dep_25 = np.reshape(np.quantile(sd_corest_dep,0.25,axis=1).T,[3*d])
corest_dep_50 = np.reshape(np.quantile(sd_corest_dep,0.50,axis=1).T,[3*d])
corest_dep_75 = np.reshape(np.quantile(sd_corest_dep,0.75,axis=1).T,[3*d])


plt.errorbar(xpts,corest_dep_50,yerr=np.array([corest_dep_50-corest_dep_25,corest_dep_75-corest_dep_50]),fmt='o')


#%% Consistency in ranking



ss_rank_cors_indep = []
kshap_rank_cors_indep = []

cv_rank_cors_indep = []
boot_rank_cors_indep = []
group_rank_cors_indep = []
wls_rank_cors_indep = []


ss_rank_cors_dep = []
kshap_rank_cors_dep = []

cv_rank_cors_dep = []
boot_rank_cors_dep = []
group_rank_cors_dep = []
wls_rank_cors_dep = []


for i in range(n_pts):
    rankmat = np.array([rankdata(sss_indep[i][j][0]) for j in range(nsim_per_point)])
    ss_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(sss_dep[i][j][0]) for j in range(nsim_per_point)])
    ss_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(sss_indep[i][j][1]) for j in range(nsim_per_point)])
    cv_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(sss_dep[i][j][1]) for j in range(nsim_per_point)])
    cv_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))  


    rankmat = np.array([rankdata(kshaps_indep[i][j][0]) for j in range(nsim_per_point)])
    kshap_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_dep[i][j][0]) for j in range(nsim_per_point)])
    kshap_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_indep[i][j][2]) for j in range(nsim_per_point)])
    boot_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_dep[i][j][2]) for j in range(nsim_per_point)])
    boot_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_indep[i][j][4]) for j in range(nsim_per_point)])
    group_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_dep[i][j][4]) for j in range(nsim_per_point)])
    group_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_indep[i][j][6]) for j in range(nsim_per_point)])
    wls_rank_cors_indep.append(np.mean(np.corrcoef(rankmat)))

    rankmat = np.array([rankdata(kshaps_dep[i][j][6]) for j in range(nsim_per_point)])
    wls_rank_cors_dep.append(np.mean(np.corrcoef(rankmat)))


