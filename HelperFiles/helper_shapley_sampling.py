from helper import *
from helper_indep import *
from helper_dep import *


def query_values_marginal(X, xloc, 
                            S, j,  mapping_dict, 
                            n_samples_per_perm):
    '''
    Strumbelj way of computing v(S) = E[f(X) | X_S]. Averaging over at least one iteration, 
    compute the model's change in prediction on xloc with and without j, 
    replacing all features in S with those of some randomly-drawn sample.
    
    In this case, we assume features are independent. We draw these features' values
    from a sample from the dataset. Technically it isn't truly from each feature's marginal distribution;
    otherwise we would draw each feature's value from a separate data sample.
    '''
    SandJ = np.append(S,j)
    n = X.shape[0]
    if mapping_dict is None:
        d = X.shape[1]
        Sc = np.where(~np.isin(np.arange(d), S))[0]
        Sjc = np.where(~np.isin(np.arange(d), SandJ))[0]
    else:
        d = len(mapping_dict)
        Sc_orig = np.where(~np.isin(np.arange(d), S))[0] # "original" low # of dimensions
        Sjc_orig = np.where(~np.isin(np.arange(d), SandJ))[0]
        Sc = map_S(Sc_orig, mapping_dict)
        Sjc = map_S(Sjc_orig, mapping_dict)

    Z_vals = []
    Zj_vals = []

    for i in range(n_samples_per_perm):
        # Sample "unknown" features from a dataset sample z zzz
        z = X[np.random.choice(n, size=1),:]
        z_x_s, z_x_s_j = np.copy(xloc), np.copy(xloc)
        z_x_s[0][Sc] = z[0][Sc]
        z_x_s_j[0][Sjc] = z[0][Sjc]

        Z_vals.append(z_x_s)
        Zj_vals.append(z_x_s_j)

    return Z_vals, Zj_vals


def query_values_conditional(X, xloc, S, j,  
                            feature_means, cov_mat, mapping_dict,
                            n_samples_per_perm):
    '''
    As above, we compute E[f(X)|X_S]. Here, though, we assume features are dependent and Gaussian.
    We sample the unknown features from their conditional Gaussian distribution: X_{-S} | X_S.
    '''
    SandJ = np.append(S,j)
    if mapping_dict is None:
        d = X.shape[1]
        Sc = np.where(~np.isin(np.arange(d), S))[0]
        Sjc = np.where(~np.isin(np.arange(d), SandJ))[0]
    else:
        d = len(mapping_dict)
        Sc_orig = np.where(~np.isin(np.arange(d), S))[0]
        Sjc_orig = np.where(~np.isin(np.arange(d), SandJ))[0]
        S = map_S(S, mapping_dict)
        SandJ = np.array(map_S(SandJ, mapping_dict))
        Sc = map_S(Sc_orig, mapping_dict)
        Sjc = map_S(Sjc_orig, mapping_dict)

    n_known_features_Sj = SandJ.shape[0]
    # Compute means and covariance of non-S features, given features in S
    means_given_S, cov_given_S = compute_cond_mean_and_cov(xloc, S, feature_means, cov_mat)
    means_given_Sj, cov_given_Sj = compute_cond_mean_and_cov(xloc, SandJ, feature_means, cov_mat)
    if cov_given_S is None or cov_given_Sj is None:
        return None, None
    
    Z_vals = []
    Zj_vals = []
    for i in range(n_samples_per_perm):
        # Sample "unknown" features from multivariate normal, conditioning on known features
        # n_samples_per_perm needs to be large enough, else f(z_x_s_j) - f(z_x_s)) will be too large
        # since more than j'th feature is different
        z_Sc = np.random.multivariate_normal(means_given_S, cov_given_S, size=1)
        if n_known_features_Sj < d:
            z_Sjc = np.random.multivariate_normal(means_given_Sj, cov_given_Sj, size=1)
            
        # Replace xloc's "unknown" features - if any - with those of random sample
        z_x_s, z_x_s_j = np.copy(xloc), np.copy(xloc)
        z_x_s[0][Sc] = z_Sc[0]
        if n_known_features_Sj < d:
            z_x_s_j[0][Sjc] = z_Sjc[0]

        Z_vals.append(z_x_s)
        Zj_vals.append(z_x_s_j)

    return Z_vals, Zj_vals

def chg_in_values(Z_vals,Zj_vals,xloc,independent_features,f_model,gradient,hessian,M,n_samples_per_perm):
    Z_vals = np.reshape(Z_vals, [M*n_samples_per_perm, xloc.shape[1]])
    Zj_vals = np.reshape(Zj_vals, [M*n_samples_per_perm, xloc.shape[1]])
    
    model_diffs = f_model(Zj_vals) - f_model(Z_vals)
    yloc = f_model(xloc)
    
    if independent_features:
        CV_diffs = f_second_order_approx(yloc,Zj_vals,xloc,gradient,hessian) - f_second_order_approx(yloc,Z_vals,xloc,gradient,hessian)
    else:
        CV_diffs = f_first_order_approx(yloc,Zj_vals,xloc,gradient) - f_first_order_approx(yloc,Z_vals,xloc,gradient)
    
    
    model_diffs = np.mean(np.reshape(model_diffs,[M,n_samples_per_perm]),axis=1)
    CV_diffs = np.mean(np.reshape(CV_diffs,[M,n_samples_per_perm]),axis=1)
    
    return model_diffs, CV_diffs


def compute_cv_shap(diffs_model, diffs_approx, shap_CV_true):
    '''
    Compute CV-based SHAP values for j'th feature, given changes in value function from M perms.
    '''
    M = len(diffs_approx)
    vanilla_shap_CV = np.mean(diffs_approx)
    vanilla_shap_model = np.mean(diffs_model)
    var_vshap_CV = np.var(diffs_approx) / M
    var_vshap_model = np.var(diffs_model) / M
    if var_vshap_CV > 0 and var_vshap_model > 0:
        # Get final estimate of the Shapley value
        cov_shap_vals = np.cov(diffs_approx, diffs_model)[0,1] / M
        alpha = cov_shap_vals / var_vshap_CV
        final_shap_est = vanilla_shap_model - alpha*(vanilla_shap_CV - shap_CV_true)
        # Compute correlation between model & approximation SHAP values
        corr = cov_shap_vals / np.sqrt(var_vshap_CV * var_vshap_model)
    else:
        final_shap_est = vanilla_shap_model
        corr = np.nan

    return final_shap_est, vanilla_shap_model, vanilla_shap_CV, corr


def cv_shapley_sampling_j(f_model, X, xloc, j,
                        independent_features,
                        shap_CV_true,
                        gradient, hessian=None,
                        mapping_dict=None,
                        M=None, t=0.05,
                        n_intermediate=50,
                        n_samples_per_perm=10, 
                        paired=True,
                        cov_mat=None,
                        verbose=False):

    '''
    Estimate SHAP value for j'th feature using our CV-based method. Main description in cv_shapley_sampling().
    '''
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    feature_means = np.mean(X, axis=0)
    if cov_mat is None:
        cov_mat = np.cov(X, rowvar=False)    
    diffs_model, diffs_approx = [], []
    count = 0
    converged = False
    
    Z_vals = []
    Zj_vals = []
    
    diffs_model = []
    diffs_approx = []
    
    while not converged:
        if not paired or count % 2 == 0:
            perm = np.random.permutation(d)
            j_idx = np.argwhere(perm==j).item()
            S = np.array(perm[:j_idx])
        else:
            # New subset is opposite of previous one
            S = perm[(j_idx+1):]
        
        if independent_features:
            # Assuming independent features, compute value the Strumbelj way:
            # Take all unknown features from a single data sample
            tZ_vals, tZj_vals = query_values_marginal(X, xloc, S, j,  
                                                            mapping_dict, n_samples_per_perm)
        else:
            # Allowing feature dependence, sample non-S features from their conditional mean.
            # Assumes multivariate normality.
            tZ_vals, tZj_vals = query_values_conditional(X, xloc, S, j,  
                                                                feature_means, cov_mat, mapping_dict,
                                                                n_samples_per_perm)
            if tZ_vals is None:
                # Could not invert matrix using sampled subset S
                continue

        Z_vals.append(tZ_vals)
        Zj_vals.append(tZj_vals)
       
        count += 1
        if M is not None:
            if count==M:
                diffs_model, diffs_approx = chg_in_values(Z_vals,Zj_vals,xloc,independent_features,f_model,gradient,hessian,M,n_samples_per_perm)
                final_shap_est, vanilla_shap_model, vanilla_shap_CV, corr = compute_cv_shap(diffs_model, diffs_approx, shap_CV_true)
                break
        elif count % n_intermediate == 0:
            
            chg_in_value_model, chg_in_value_approx = chg_in_values(Z_vals,Zj_vals,xloc,independent_features,f_model,gradient,hessian,n_intermediate,n_samples_per_perm)
            
            diffs_model.append(chg_in_value_model)
            diffs_approx.append(chg_in_value_approx)

            Z_vals = []
            Zj_vals = []
            
            # Check for convergence - when estimated SHAP values aren't so variable relative to their magnitudes
            final_shap_est, vanilla_shap_model, vanilla_shap_CV, corr = compute_cv_shap(diffs_model, diffs_approx, shap_CV_true)
            
            if corr < 1: # Sometimes it's greater (rounding error?)
                var_vshap_model = np.var(diffs_model) / count
                var_final_shap = (1 - corr**2)*var_vshap_model
                prop_std = np.sqrt(var_final_shap) / np.abs(final_shap_est)
                if prop_std < t:
                    converged = True
                    if verbose:
                        print("Converged with {} samples.".format(count))

    final_shap_est = final_shap_est.item()
    return final_shap_est, vanilla_shap_model, vanilla_shap_CV, corr


def cv_shapley_sampling(f_model, X, xloc, 
                        independent_features,
                        gradient, hessian=None,
                        shap_CV_true=None,
                        M=100, n_samples_per_perm=10, 
                        mapping_dict=None,
                        cov_mat=None, D_matrices=None,
                        t=0.05, n_intermediate=50,
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
    - D_matrices: Matrices used to compute true SHAP value of control variate in dependent features case.
            Required if independent_features=False and shap_CV_true=None.
    - t: Stopping threshold if M not specified. From Covert & Lee:
            Stop when the largest standard deviation is a sufficiently
            small portion t (e.g., t = 0.01) of the gap between
            the largest and smallest Shapley value estimates.
    - n_intermediate: Regularity with which to check whether convergence has occurred. If M not specified.
    - paired: Bool for whether to do paired sampling, from Covert & Lee. 
            Pair each random permutation with its complement. 
            Should be good in general, but won't impact CV-adjustment
    - verbose: Bool, whether to say speed of convergence. Used if M not specified.

    Outputs:
    A tuple of 4 elements: 
        The CV-adjusted SHAP estimates
        The vanilla SHAP estimates of the model
        The vanilla SHAP estimates of the approximation
        The correlations between the SHAP estimates of the model & approximation 
    '''
    if independent_features and hessian is None:
        sys.exit("Need to provide hessian matrix in independent features case.")
    feature_means = np.mean(X, axis=0)
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    if cov_mat is None:
        cov_mat = np.cov(X, rowvar=False)
    if shap_CV_true is None:
        if independent_features:
            shap_CV_true = compute_true_shap_cv_indep(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict)
        else: # Dependent features
            if D_matrices is None:
                sys.exit("In dependent features case, need to input either SHAP values of control variate (shap_CV_true) or matrices used to estimate them (D_matrices).")
            shap_CV_true = linear_shap_vals(xloc, D_matrices, feature_means, gradient, mapping_dict)
    
    obj = [cv_shapley_sampling_j(f_model, X, xloc, j,
                        independent_features,
                        shap_CV_true=shap_CV_true[j],
                        gradient=gradient, hessian=hessian,
                        mapping_dict=mapping_dict,
                        M=M, t=t,
                        n_intermediate=n_intermediate,
                        n_samples_per_perm=n_samples_per_perm, 
                        paired=paired,
                        cov_mat=cov_mat,
                        verbose=verbose) for j in range(d)]
    obj = np.array(obj).T
    return obj[0], obj[1], obj[2], obj[3]

