import numpy as np
from helper import *

"""
Functions relevant for the dependent features case.
"""


def f_first_order_approx(yloc, xnew, xloc, gradient):
    '''
    First-order (linear) approximation to model f around xloc. 
    Relevant when assuming feature dependence for both SHAP and kernelSHAP.
    '''
    if xnew.ndim==1:
        xnew = xnew.reshape((1,xnew.shape[0]))

    yloc_pred = yloc
    n, d = xnew.shape
    if n==1:
        deltaX = np.array(xnew - xloc).reshape((d, -1))
        first_order_approx = yloc_pred + np.dot(deltaX.T, gradient)
        return first_order_approx.item()
    else:
        first_order_approx = np.zeros(n)
        for i in range(n):
            deltaX_i = np.array(xnew[i,:] - xloc).reshape((d, -1))
            first_order_approx[i] = yloc_pred + np.dot(deltaX_i.T, gradient)
        return first_order_approx


def make_banded_cov(FEATURE_VARS, FEATURE_COVS):
    '''
    Makes banded covariance matrix for simulated data. 
    - feature_vars contains the diagonal variances
    - feature_covs is a list of two off-diagonal covariances, e.g. 0.5 and 0.25.
    Then covariance between successive features is 0.5
        e.g. Cov(X1, X2)=0.5; Cov(X2, X3)=0.5; ...
    Similarly, Cov(X_j, X_{j+2}) = 0.25 for j in 0, ..., d-3
    '''
    d = FEATURE_VARS.shape[0]
    cov_mat = np.identity(d) * FEATURE_VARS
    for j in range(d):
        if j+1 <= d-1:
            cov_mat[j, j+1] = FEATURE_COVS[0]
        if j-1 >= 0:
            cov_mat[j, j-1] = FEATURE_COVS[0]
        if j+2 <= d-1:
            cov_mat[j, j+2] = FEATURE_COVS[1]
        if j-2 >= 0:
            cov_mat[j, j-2] = FEATURE_COVS[1]
    return cov_mat


def compute_cond_mean_and_cov(xloc, S, feature_means, cov_mat):
    '''
    Given a subset S of features from d-dimensional data X,
    compute the mean and covariance of X_{S^c} | X_S.
    '''
    d = feature_means.shape[0]
    Sc = np.where(~np.isin(np.arange(d), S))[0]
    cov_SS = cov_mat[S][:, S]
    try:
        cov_SS_inv = np.linalg.inv(cov_SS) # Sometimes triggers warning
        cov_ScS = cov_mat[Sc][:, S]
        conditional_means = feature_means[Sc] + np.matmul(np.matmul(cov_ScS, cov_SS_inv), xloc[0][S] - feature_means[S])
        cov_ScSc = cov_mat[Sc][:, Sc]
        conditional_cov = cov_ScSc - np.matmul(np.matmul(cov_ScS, cov_SS_inv), cov_ScS.T)
        return conditional_means, conditional_cov
    except:
        return None, None

## Functions used for computing SHAP values given feature dependence.
def get_S_complement(S, d):
    Sc = np.where(~np.isin(np.arange(d), S))[0]
    return Sc

def make_proj_mat(S, d):
        # Want the "projection matrix that selects a set"
        subset_size = S.shape[0]
        P_S = np.zeros((subset_size, d))
        for i in range(subset_size):
            P_S[i,S[i]] = 1
        return P_S

def make_Q_S(S, d):
    P_S = make_proj_mat(S, d)
    Q_S = P_S.T @ P_S
    return Q_S

def make_R_S(S, d, cov_mat):
    P_S = make_proj_mat(S, d)
    Sc = get_S_complement(S, d)
    P_Sc = make_proj_mat(Sc, d)
    term1 = (P_Sc.T @ P_Sc) @ cov_mat @ P_S.T
    term2 = np.linalg.inv(P_S @ cov_mat @ P_S.T) @ P_S
    R_S = term1 @ term2
    return R_S


def make_all_lundberg_matrices(M_linear, cov_mat):
    '''
    Computationally intensive work necessary for estimating SHAP value of linear model w dependent features.
    Creates matrices D^j for all features.
    '''
    def make_jth_lundberg_matrix(M, j, cov_mat):
        d = cov_mat.shape[0]
        D_j = np.zeros((d,d))
        ct = 0
        while ct < M:
            try:
                perm = np.random.permutation(d)
                j_idx = np.argwhere(perm==j).item()
                S = perm[:j_idx]

                Q_S = make_Q_S(S, d)
                R_S = make_R_S(S, d, cov_mat)
                SandJ = np.append(S, j)
                Q_SandJ = make_Q_S(SandJ, d)
                R_SandJ = make_R_S(SandJ, d, cov_mat)
                D_S_j = (Q_SandJ + R_SandJ) - (Q_S + R_S)
                D_j += D_S_j
                ct += 1
            except:
                continue
        D_j = D_j / M
        return D_j
        
    d = cov_mat.shape[0]
    D_matrices = [make_jth_lundberg_matrix(M_linear, j, cov_mat) for j in range(d)]
    return D_matrices

def linear_shap_j(xloc, C_j, D_j, feature_means, gradient):
    d = xloc.shape[1]
    coefs = gradient.reshape((1, d)) # slope of linear approximation
    shap_j = (coefs @ C_j @ feature_means + coefs @ D_j @ xloc[0]).item()
    return(shap_j)

def linear_shap_vals(xloc, D_matrices, feature_means, gradient, mapping_dict=None):
    '''
    Returns set of SHAP values for all features in linear approximation.
    If mapping_dict is not none, sums SHAP values of all binary columns 
        corresponding to an individual multilevel categorical feature.
    '''
    d = xloc.shape[1]
    coefs = gradient.reshape((1, d)) # slope of linear approximation
    all_shap_vals = np.array([(coefs @ D_matrices[j] @ (xloc[0] - feature_means)).item() for j in range(d)])
    if mapping_dict is None:
        return all_shap_vals
    else:
        # Account for multilevel features
        shap_CV_mapped = []
        d_orig = len(mapping_dict)
        for i in range(d_orig):
            relevant_cols = mapping_dict[i]
            if len(relevant_cols)==1: # Not a column of a multilevel feature
                shap_CV_mapped.append(all_shap_vals[relevant_cols].item())
            else:
                shap_CV_mapped.append(np.sum(all_shap_vals[relevant_cols]))
        shap_CV_mapped = np.array(shap_CV_mapped)
        return(shap_CV_mapped)

