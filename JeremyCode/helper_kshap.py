import numpy as np
import sys
from helper import *
from helper_indep import *
from helper_dep import *
from math import comb


def compute_avg_preds(f_model, X, xloc, independent_features, gradient, hessian=None):
    '''
    Computes the average prediction of the model and its linear or quadratic approximation.
    '''
    avg_pred_model = np.mean(f_model(X))
    if independent_features:
        avg_pred_CV = np.mean(f_second_order_approx(f_model,X, xloc, gradient, hessian))
    else:
        avg_pred_CV = np.mean(f_first_order_approx(f_model, X, xloc, gradient))
    return avg_pred_model, avg_pred_CV


def conditional_means_kshap(f_model, X, xloc, z, 
                            independent_features, 
                            feature_means, cov_mat, 
                            n_samples_per_perm,
                            gradient, hessian=None,
                            mapping_dict=None):
    '''
    Computes E[f(X)|X_S] for both f: the model and its approximation.
    z is a "coalition" vector whose nonzero (1) entries denote S, the features we condition on.
    '''

    # Of true num features, if mapping_dict not None
    d = z.shape[0] 
    if mapping_dict is None:
        S = np.nonzero(z)[0]
        Sc = np.where(~np.isin(np.arange(d), S))[0]
    else:
        # "original" low # of dimensions
        S_orig = np.nonzero(z)[0]
        Sc_orig = np.where(~np.isin(np.arange(d), S_orig))[0]
        # High # of dimensions (each binary level as a column)
        S = map_S(S_orig, mapping_dict)
        Sc = map_S(Sc_orig, mapping_dict)
    
    # Uses same data sample(s) for both black-box model f and approximation f
    preds_given_S_model = []
    preds_given_S_CV = []
    if not independent_features: 
        # Sample non-S features given S from conditional (normal) distribution, not marginal
        means_given_S, cov_given_S = compute_cond_mean_and_cov(xloc, S, feature_means, cov_mat)

    for i in range(n_samples_per_perm):
        if independent_features:# Which features are known is irrelevant
            # data_sample = np.random.multivariate_normal(feature_means, cov_mat, size=1)
            z = X[np.random.choice(X.shape[0], size=1),:]
        else:
            z_Sc = np.random.multivariate_normal(means_given_S, cov_given_S, size=1)
    
        # Copy xloc, then replace its "unknown" features with those of random sample z
        z_x_s = np.copy(xloc)
        if not independent_features:
            z_x_s[0][Sc] = z_Sc[0]
        else:
            z_x_s[0][Sc] = z[0][Sc]

        preds_given_S_model.append(f_model(z_x_s))
        if independent_features:
            preds_given_S_CV.append(f_second_order_approx(f_model, z_x_s, xloc, gradient, hessian))
        else:
            preds_given_S_CV.append(f_first_order_approx(f_model, z_x_s, xloc, gradient))
    value_z_model = np.mean(preds_given_S_model)
    value_z_CV = np.mean(preds_given_S_CV)
    return value_z_model, value_z_CV


def cv_kshap(f_model, X, xloc, 
            independent_features,
            gradient, hessian=None,
            shap_CV_true=None,
            M=1000, n_samples_per_perm=10,
            mapping_dict=None,
            cov_mat=None, 
            var_method='boot', n_boot=250,  
            K=50, D_matrices=None, 
            t=0.1, n_intermediate=50,
            paired=True, 
            verbose=False):
    '''
    Estimates all SHAP values of the model and its approximation, then adjusts with the control variate.
    Inputs:
    - f_model: which inputs a numpy array and outputs a scalar
    - X: the entire dataset
    - xloc: the "local" x about which we calculate the Shapley values
    - independent_features: a bool designating whether we use a linear or quadratic approxmation.
    - gradient: The gradient of f_model with respect to xloc.
    - hessian: The hessian of f_model with respect to xloc. Used if independent_features=True.
    - shap_CV_true: The vector of true SHAP values of the model approximation. This is our control variate.
    - M: The number of permutations
    - n_samples_per_perm: The number of samples used to compute Ef(X)|X_S for each S
    - mapping_dict: A dictionary linking feature indices to columns. 
            Need if there are categorical features with more than 2 levels.
            Each key is an index of the number of "real" features in the dataset.
            Each value is a list of indices reflecting which columns in X correspond to that index's feature.
    - cov_mat: Covariance matrix. If none, we compute naive way; risks poor conditioning.
    - var_method: Which method to use to estimate variance and covariance of SHAP estimates.
            Must be 'boot', 'grouped', or 'wls'. 'boot' or 'wls' recommended.
    - n_boot: Number of samples for bootstrapping. Used if var_method is 'boot'.
    - K: Size of each group for computing SHAP estimates. Used if var_method is 'grouped'.
    - D_matrices: Matrices used to compute true SHAP value of control variate in dependent features case.
            Required if independent_features=False and shap_CV_true=None.
    - t: Stopping threshold if M not specified. From Covert & Lee:
            Stop when the largest standard deviation is a sufficiently
            small portion t (e.g., t = 0.01) of the gap between
            the largest and smallest Shapley value estimates.
    - n_intermediate: Regularity with which to check whether convergence has occurred. If M not specified.
    - paired: Bool for whether to do paired sampling, from Covert & Lee. 
            Pair each random permutation with its complement. 
    - verbose: Bool, whether to say speed of convergence. Used if M not specified.

    Outputs a tuple of 4 elements: 
        The CV-adjusted SHAP estimates
        The vanilla SHAP estimates of the model
        The vanilla SHAP estimates of the approximation
        The correlations between the SHAP estimates of the model & approximation 
    '''

    converged = False
    count = 0
    # n, d = X.shape
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    avg_pred_model, avg_pred_CV = compute_avg_preds(f_model, X, xloc, independent_features, gradient, hessian)

    kernel_weights = [0]*(d+1)
    for subset_size in range(d+1):
        if subset_size > 0 and subset_size < d:
            kernel_weights[subset_size] = (d-1)/(comb(d,subset_size)*subset_size*(d-subset_size))
    subset_size_distr = np.array(kernel_weights) / np.sum(kernel_weights)

    feature_means = np.mean(X, axis=0)
    if cov_mat is None:
        cov_mat = np.cov(X, rowvar=False)
    if shap_CV_true is None:
        # Compute (or closely estimate) true SHAP Values of control variate
        if independent_features:
            shap_CV_true = compute_true_shap_cv_indep(xloc, gradient, hessian, feature_means, cov_mat)
        else:
            if D_matrices is None:
                sys.exit("In dependent features case, need to input either SHAP values of control variate (shap_CV_true) or matrices used to estimate them (D_matrices).")
            shap_CV_true = linear_shap_vals(xloc, D_matrices, feature_means, gradient, mapping_dict)

    coalitions = []
    coalition_values_model = []
    coalition_values_CV = []
    while not converged:
        if not paired or count % 2 == 0:
            subset_size = np.random.choice(np.arange(len(subset_size_distr)), p=subset_size_distr)
            # Randomly choose these features, then convert to binary vector
            S = np.random.choice(d, subset_size, replace=False)
            z = np.zeros(d)
            z[S] = 1
        else:
            # New coalition is inverse of previous one
            z = 1 - z

        # Estimate v(z_S) = E[f(X) | X_S = x_S]
        value_z_model, value_z_CV = conditional_means_kshap(f_model, X, xloc, z, 
                            independent_features, 
                            feature_means, cov_mat, 
                            n_samples_per_perm,
                            gradient, hessian,
                            mapping_dict)

        count += 1
        coalition_values_model = np.append(coalition_values_model, value_z_model)
        coalition_values_CV = np.append(coalition_values_CV, value_z_CV)
        # Append new coalition vector (feature-dim if mapped)
        coalitions = np.append(coalitions, z).reshape((count, d))

        if M is not None:
            if count==M:
                # Compute vanilla kernelSHAP estimate for model and control variate
                vanilla_kshap_model = kshap_equation(f_model, xloc, coalitions, coalition_values_model, avg_pred_model)
                vanilla_kshap_CV = kshap_equation(f_model, xloc, coalitions, coalition_values_CV, avg_pred_CV)
                # Compute variance and covariance estimates
                n_boot = min(M, n_boot)
                kshap_vars_model, kshap_vars_CV, kshap_covs = kshap_vars_and_cov(f_model, xloc,
                                                                    avg_pred_model, avg_pred_CV, 
                                                                    coalitions, coalition_values_model, 
                                                                    coalition_values_CV,
                                                                    vanilla_kshap_model, vanilla_kshap_CV,
                                                                    var_method, n_boot,  K, subset_size_distr)
                # Compute CV-based KernelSHAP estimate
                final_kshap_ests, corrs = final_cv_kshap(vanilla_kshap_model, vanilla_kshap_CV, shap_CV_true,
                                kshap_covs, kshap_vars_model, kshap_vars_CV)
                converged = True
        elif count % n_intermediate == 0:
            try:
                # Compute vanilla SHAP values for model & approximation
                vanilla_kshap_model = kshap_equation(f_model, xloc, coalitions, coalition_values_model, avg_pred_model)
                vanilla_kshap_CV = kshap_equation(f_model, xloc, coalitions, coalition_values_CV, avg_pred_CV)

                # Compute variance & covariance of these SHAP estimates. 
                # wls variance method doesn't work & grouped method only works when count is large, so bootstrapping.
                n_boot_ = min(count, n_boot)
                kshap_model, kshap_CV = boot_cv_kshap(f_model, xloc, avg_pred_model, avg_pred_CV, coalitions, coalition_values_model, coalition_values_CV, n_boot_)
                kshap_vars_CV = compute_kshap_vars_boot(kshap_CV)
                kshap_vars_model = compute_kshap_vars_boot(kshap_model)
                kshap_covs = compute_kshap_covs_boot(kshap_model, kshap_CV)
                
                # Compute final kSHAP estimate
                final_kshap_ests, corrs = final_cv_kshap(vanilla_kshap_model, vanilla_kshap_CV, shap_CV_true,
                                kshap_covs, kshap_vars_model, kshap_vars_CV)
                # Stop once the variability of the SHAP values is minimal relative to their spread.  
                if np.max(corrs) < 1:
                    kshap_vars_final = [(1 - corrs[j]**2)*kshap_vars_model[j] for j in range(d)]
                    max_std_dev = np.sqrt(np.max(kshap_vars_final))
                    prop_std = max_std_dev / (np.max(final_kshap_ests) - np.min(final_kshap_ests))
                    if prop_std < t:
                        converged = True
                        if verbose:
                            print("Converged with {} samples.".format(count))
            except:
                continue
    return final_kshap_ests, vanilla_kshap_model, vanilla_kshap_CV, corrs


def kshap_vars_and_cov(f_model, xloc,
                        avg_pred_model, avg_pred_CV, 
                        coalitions, coalition_values_model, 
                        coalition_values_CV,
                        vanilla_kshap_model, vanilla_kshap_CV,
                        var_method, n_boot,  K, subset_size_distr):
    if var_method=='boot':
        kshap_model, kshap_CV = boot_cv_kshap(f_model, xloc, avg_pred_model, avg_pred_CV, coalitions, coalition_values_model, coalition_values_CV, n_boot)
        kshap_vars_CV = compute_kshap_vars_boot(kshap_CV)
        kshap_vars_model = compute_kshap_vars_boot(kshap_model)
        kshap_covs = compute_kshap_covs_boot(kshap_model, kshap_CV)
    elif var_method=='grouped':
        kshap_model, kshap_CV = grouped_cv_kshap(f_model, xloc, avg_pred_model, avg_pred_CV, coalitions, coalition_values_model, coalition_values_CV, K)
        kshap_vars_CV = compute_kshap_vars_grouped(kshap_CV, K)
        kshap_vars_model = compute_kshap_vars_grouped(kshap_model, K)
        kshap_covs = compute_kshap_covs_grouped(kshap_model, kshap_CV, K)
    elif var_method=='wls':
        kshap_vars_CV = compute_kshap_vars_wls(coalitions, coalition_values_CV, vanilla_kshap_CV, avg_pred_CV, subset_size_distr)
        kshap_vars_model = compute_kshap_vars_wls(coalitions, coalition_values_model, vanilla_kshap_model, avg_pred_model, subset_size_distr)
        kshap_covs = compute_kshap_covs_wls(coalitions, coalition_values_model, coalition_values_CV, subset_size_distr)
    else:
        sys.exit("var_method needs to be boot, grouped, or wls.")
    return kshap_vars_model, kshap_vars_CV, kshap_covs


def kshap_equation(f_model, xloc, coalitions, coalition_values, avg_pred):
    '''
    Computes KernelSHAP estimates for all features. The equation is the solution to the 
    weighted least squares problem of KernelSHAP. This inputs the dataset of M (z, v(z)).

    If multilevel, coalitions is binary 1s & 0s of the low-dim problem.
    '''
    
    # Compute v(1), the prediction made using all known features in xloc
    # d = xloc.shape[1]
    d = coalitions.shape[1] # low-dim if mapped
    M = coalition_values.shape[0]
    yloc_pred = f_model(xloc) # True even if using approximation not black-box model
    avg_pred_vec = np.repeat(avg_pred, M)

    # A matrix and b vector in Covert and Lee
    A = np.matmul(coalitions.T, coalitions) / M
    b = np.matmul(coalitions.T, coalition_values - avg_pred_vec) / M

    # Covert & Lee Equation 7
    try:
        A_inv = np.linalg.inv(A)
    except:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0]/new_cond_num
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s2), vh))

        A_inv = np.linalg.inv(A2)
    ones_vec = np.ones(d).reshape((d, 1))
    numerator = np.matmul(np.matmul(ones_vec.T, A_inv), b) - yloc_pred + avg_pred
    denominator = np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)
    term = (b - (numerator / denominator)).reshape((d, 1))

    kshap_ests = np.matmul(A_inv, term).reshape(-1)
    return kshap_ests


def boot_cv_kshap(f_model, xloc, avg_pred_model, avg_pred_CV, coalitions, 
                    coalition_values_model, coalition_values_CV, n_boot):
    """
    Returns n_boot sets of kernelSHAP values for each feature, 
    fitting kernelSHAP on both the true model and its approximation with each bootstrapped resampling.

    We can probably make a version of this function where you don't have to bootstrap the model,
    since we don't need this for computing CV-kSHAP estimates; we only need it to compute the
    variance reduction of CV-kSHAP over vanilla kSHAP. Low priority.
    """

    kshap_model_boot_all = []
    kshap_CV_boot_all = []
    M = coalition_values_CV.shape[0]
    for _ in range(n_boot):
        # sample M (z, v(z)) pairs with replacement. this will be our coalition dataset.
        # If paired, bootstrapped sample will include pairs (S, S^c)
        # idx = np.random.randint(M, size=M)
        even_idx = np.random.randint(M//2, size=M//2)*2
        idx = np.concatenate((even_idx, even_idx + 1))
        z_boot = coalitions[idx]
        coalition_values_CV_boot = coalition_values_CV[idx]
        coalition_values_model_boot = coalition_values_model[idx]

        # compute the kernelSHAP estimates on these bootstrapped samples, fitting WLS
        kshap_CV_boot_all.append(kshap_equation(f_model, xloc, z_boot, coalition_values_CV_boot, avg_pred_CV))
        kshap_model_boot_all.append(kshap_equation(f_model, xloc, z_boot, coalition_values_model_boot, avg_pred_model))

    kshap_CV_boot_all = np.stack(kshap_CV_boot_all, axis=0)
    kshap_model_boot_all = np.stack(kshap_model_boot_all, axis=0)
    return kshap_model_boot_all, kshap_CV_boot_all


def compute_kshap_vars_boot(kshap_boot):
    """
    Computes the empirical variance of each feature's KernelSHAP value, using bootstrapped samples.
    """

    kshap_vars_boot = np.cov(np.array(kshap_boot), rowvar=False)
    return np.diag(kshap_vars_boot)


def grouped_cv_kshap(f_model, xloc, avg_pred_model, avg_pred_CV, coalitions, coalition_values_model, coalition_values_CV, K=50):
    '''
    Returns M/K sets of model & approximation kernelSHAP values for each feature. Each set corresponds to a different 
    non-overlapping subgroup of size K of the M sampled pairs. 
    '''
    M = coalition_values_CV.shape[0]
    n_groups = int(M/K) # floor division
    kshap_CV_grouped = []
    kshap_model_grouped = []
    for i in range(n_groups):
        # Get K <z, v(z)> pairs for each group
        idx = np.arange(start=K*i, stop=K*(i+1))
        coalition_group = coalitions[idx]
        coalition_values_CV_group = coalition_values_CV[idx]
        coalition_values_model_group = coalition_values_model[idx]

        # compute the kernelSHAP estimates on these bootstrapped samples, fitting WLS
        kshap_CV_grouped.append(kshap_equation(f_model, xloc, coalition_group, coalition_values_CV_group, avg_pred_CV))
        kshap_model_grouped.append(kshap_equation(f_model, xloc, coalition_group, coalition_values_model_group, avg_pred_model))

    kshap_CV_grouped = np.stack(kshap_CV_grouped, axis=0)
    kshap_model_grouped = np.stack(kshap_model_grouped, axis=0)
    return kshap_model_grouped, kshap_CV_grouped


def compute_kshap_vars_grouped(kshap_ests_grouped, K=50):
    '''
    Estimates the variance of each feature's kernelSHAP value, using the grouped method.
    '''
    M = K * kshap_ests_grouped.shape[0]
    kshap_vars_subgroup = np.cov(kshap_ests_grouped, rowvar=False)
    return (K/M)*np.diag(kshap_vars_subgroup)


def compute_kshap_vars_wls(coalitions, coalition_values, kshap_ests, avg_pred, subset_size_distr):
    M = coalition_values.shape[0]
    mean_subset_values = np.matmul(coalitions, kshap_ests) + avg_pred
    var_values = np.mean((coalition_values - mean_subset_values)**2) * np.identity(M) 
    counts = np.sum(coalitions, axis=1).astype(int).tolist()
    W = np.diagflat([subset_size_distr[counts[i]] for i in range(M)])
    inv_ZTW = np.linalg.inv(coalitions.T @ W @ coalitions) @ coalitions.T @ W
    kshap_vars_CV_wls = np.diagonal(inv_ZTW @ var_values @ inv_ZTW.T)
    return kshap_vars_CV_wls


def compute_kshap_covs_boot(kshap_model_boot_all, kshap_CV_boot_all):
    '''
    On each feature, estimates the covariance of its kernelSHAP values for the original model and its approximation.
    Does so using bootstrapped samples.
    '''
    d = kshap_model_boot_all.shape[1]
    kshap_covs_boot = [np.cov(kshap_CV_boot_all[:,j], kshap_model_boot_all[:,j])[0,1] for j in range(d)]
    return kshap_covs_boot


def compute_kshap_covs_grouped(kshap_model_grouped, kshap_CV_grouped, K):
    '''
    On each feature, estimates the covariance of its kernelSHAP values for the original model and its approximation.
    Does so using grouped method.
    '''
    d = kshap_CV_grouped.shape[1]
    M = K * kshap_CV_grouped.shape[0]
    kshap_covs_grouped = [(K/M) * np.cov(kshap_CV_grouped[:,j], kshap_model_grouped[:,j])[0,1] for j in range(d)]
    return kshap_covs_grouped

def compute_kshap_covs_wls(coalitions, coalition_values_model, coalition_values_CV, subset_size_distr):
    M = coalition_values_model.shape[0]
    cov_values = np.cov(coalition_values_model, coalition_values_CV)[0,1] * np.identity(M)
    # W = np.diag(subset_size_distr[np.sum(coalitions, axis=0)])
    counts = np.sum(coalitions, axis=1).astype(int).tolist()
    W = np.diagflat([subset_size_distr[counts[i]] for i in range(M)])
    inv_ZTW = np.linalg.inv(coalitions.T @ W @ coalitions) @ coalitions.T @ W
    kshap_covs_CV_wls = np.diagonal(inv_ZTW @ cov_values @ inv_ZTW.T)
    return kshap_covs_CV_wls


def final_cv_kshap(vanilla_kshap_model, vanilla_kshap_CV, shap_CV_true,
                    kshap_covs, kshap_vars_model, kshap_vars_CV):
        d = vanilla_kshap_model.shape[0]
        final_kshap_ests = np.zeros(d)
        corrs = np.zeros(d)
        for j in range(d):
            # Compute and save our method's kernelSHAP estimate
            alpha_j = kshap_covs[j]/kshap_vars_CV[j]
            final_kshap_ests[j] = vanilla_kshap_model[j] - alpha_j*(vanilla_kshap_CV[j] - shap_CV_true[j])
            # auxiliary
            corrs[j] = kshap_covs[j] / np.sqrt(kshap_vars_model[j] * kshap_vars_CV[j])
        return final_kshap_ests, corrs

