import numpy as np
from helper import *

"""
Functions relevant for the independent features case.
"""

def f_second_order_approx(f_model, xnew, xloc, gradient, hessian):
    '''
    Second order approximation to model at xnew, a point around xloc. 
    Relevant when assuming feature independence for both SHAP and kernelSHAP.
    '''
    if xnew.ndim==1:
        xnew = xnew.reshape((1,xnew.shape[0]))
    
    yloc_pred = f_model(xloc)
    n, d = xnew.shape
    if n==1:
        deltaX = np.array(xnew - xloc).reshape((d, -1))
        second_order_approx = yloc_pred + np.dot(deltaX.T, gradient) + 0.5*np.dot(np.dot(deltaX.T, hessian), deltaX) 
        return second_order_approx.item()
    else:
        second_order_approx = np.zeros(n)
        for i in range(n):
            deltaX_i = np.array(xnew[i,:] - xloc).reshape((d, -1))
            second_order_approx[i] = yloc_pred + np.dot(deltaX_i.T, gradient) + 0.5*np.dot(np.dot(deltaX_i.T, hessian), deltaX_i)
        return second_order_approx


def compute_true_shap_cv_indep(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict=None):
    '''
    Computes exact Shapley value of control variate in independent features case (second-order approximation).
    '''
    def compute_jth_shap_val(xloc, feature_means, cov_mat, j, gradient, hessian):
        d = xloc.shape[1]
        mean_j = feature_means[j]
        xloc_j = xloc[0,j]
        linear_term = gradient[j]*(xloc_j - mean_j)
        mean_term = -0.5*(mean_j - xloc_j) * np.sum([(feature_means[k]-xloc[0,k])*hessian[j,k] for k in range(d)])
        var_term = -0.5*np.sum([cov_mat[j,k]*hessian[j,k] for k in range(d)])
        # old_var_term = -0.5*cov_mat[j,j]*hessian[j,j]
        jth_shap_val = linear_term + mean_term + var_term
        return jth_shap_val

    d_total = xloc.shape[1]
    shap_vals = np.array([compute_jth_shap_val(xloc, feature_means, cov_mat, j, gradient, hessian) for j in range(d_total)])

    if mapping_dict is None:
        return shap_vals
    else:
        # Account for multilevel features
        shap_CV_true = []
        d = len(mapping_dict)
        for i in range(d):
            relevant_cols = mapping_dict[i]
            if len(relevant_cols)==1: # Not a column of a multilevel feature
                shap_CV_true.append(shap_vals[relevant_cols].item())
            else:
                shap_CV_true.append(np.sum(shap_vals[relevant_cols]))

        return np.array(shap_CV_true)
        
