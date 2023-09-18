import numpy as np
from helper2 import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *

def fullsim(fname,X_locs,X,model,gradfn,hessfn,D_matrices,mapping_dict=[],sds=[],
            nsim_per_point=50,M=1000,n_samples_per_perm=10,K=50, n_boot=100):
    
    n_pts = X_locs.shape[0]
    d = X.shape[1]
    
    feature_means = np.mean(X, axis=0)
    cov_mat = np.cov(X, rowvar=False)
    
    kshaps_indep = []
    kshaps_dep = []
    sss_indep = []
    sss_dep = []
    
    for i in range(n_pts):
        xloci = X_locs[i].reshape((1,d))
        
        gradient = gradfn(model,xloci,sds)
        hessian = hessfn(model,xloci,sds)
        
        shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat, mapping_dict)
        shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient,mapping_dict=mapping_dict)
        
        sims_kshap_indep = []
        sims_kshap_dep = []
        sims_ss_indep = []
        sims_ss_dep = []
        for j in range(nsim_per_point):
            print([i,j])
            independent_features=True
            try:
                obj_kshap_indep = cv_kshap_compare(model, X, xloci,
                                    independent_features,
                                    gradient, hessian,
                                    shap_CV_true=shap_CV_true_indep,
                                    M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot,
                                    mapping_dict=mapping_dict)        
            except:
                print('kshap indep exception')
                obj_kshap_indep = []
                for _ in range(8):
                    obj_kshap_indep.append( np.repeat(float('nan'),len(shap_CV_true_indep)))
                        
            sims_kshap_indep.append(obj_kshap_indep)
            
            try:
                obj_ss_indep = cv_shapley_sampling(model, X, xloci, 
                                        independent_features,
                                        gradient, hessian, shap_CV_true=shap_CV_true_indep,
                                        M=M, n_samples_per_perm=n_samples_per_perm,
                                        mapping_dict=mapping_dict)
            except:
                 print('ss indep exception')
                 obj_ss_indep = []
                 for _ in range(4):
                     obj_ss_indep.append( np.repeat(float('nan'),len(shap_CV_true_indep)))
                     
            sims_ss_indep.append(obj_ss_indep)
            
            independent_features=False
            try:
                obj_kshap_dep = cv_kshap_compare(model, X, xloci,
                                    independent_features,
                                    gradient,
                                    shap_CV_true=shap_CV_true_dep,
                                    M=M, n_samples_per_perm=n_samples_per_perm, K = K, n_boot=n_boot, 
                                    cov_mat=cov_mat, 
                                    mapping_dict=mapping_dict)
            except:
                print('kshap dep exception')
                obj_kshap_dep = []
                for _ in range(8):
                    obj_kshap_dep.append( np.repeat(float('nan'),len(shap_CV_true_dep)))
                    
            sims_kshap_dep.append(obj_kshap_dep)
            
            try:
                obj_ss_dep = cv_shapley_sampling(model, X, xloci, 
                                        independent_features,
                                        gradient, shap_CV_true=shap_CV_true_dep,
                                        M=M, n_samples_per_perm=n_samples_per_perm, cov_mat=cov_mat,
                                        mapping_dict=mapping_dict)
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
    
        np.save(fname+'_kshap_indep.npy',kshaps_indep)
        np.save(fname+'_kshap_dep.npy',kshaps_dep)
        np.save(fname+'_ss_indep.npy',sss_indep)
        np.save(fname+'_ss_dep.npy',sss_dep)