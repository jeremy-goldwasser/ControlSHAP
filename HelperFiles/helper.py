import sys
import numpy as np
import matplotlib.pyplot as plt


"""
Functions relevant for independent- and dependent-features, 
Shapley Sampling and kernelSHAP
"""
def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return 1 / (1 + np.exp(-x.item()))

def logreg_gradient(f_model, xloc, BETA):
    '''
    Computes the gradient of a logistic regression function f_model with parameters BETA at xloc.
    '''
    yloc_pred = f_model(xloc)
    d = BETA.shape[0]
    return np.array(yloc_pred*(1-yloc_pred)*BETA).reshape((d, 1))


def logreg_hessian(f_model, xloc, BETA):
    '''
    Computes the hessian of a logistic regression function f_model with parameters BETA at xloc.
    '''
    yloc_pred = f_model(xloc)
    d = BETA.shape[0]
    beta_2d = np.array(BETA).reshape((d, -1))
    BBT = np.dot(beta_2d, beta_2d.T)
    return yloc_pred*(1-yloc_pred)*(1-2*yloc_pred)*BBT


def extract_results(obj):
    '''
    Often, we'll have run cv_kshap() or cv_shapley_sampling() many times.
    This function aggregates all their Shapley estimates and correlations
    into four numpy arrays.
    '''

    n_iter = len(obj)
    final_shap_ests = np.array([obj[i][0] for i in range(n_iter)])
    vanilla_shap_model = np.array([obj[i][1] for i in range(n_iter)])
    vanilla_shap_CV = np.array([obj[i][2] for i in range(n_iter)])
    corr_ests = np.array([obj[i][3] for i in range(n_iter)])
    return final_shap_ests, vanilla_shap_model, vanilla_shap_CV, corr_ests


def show_var_reducs(obj, verbose=False, message=None, ylim_zero=False):
    '''
    Displays theoretical and empirical variance reductions of our method.
    obj is a list of many runnings of cv_kshap() or cv_shapley_sampling().

    '''
    final_ests, vshap_ests_model, vshap_ests_CV, corr_ests = extract_results(obj)
    d = final_ests.shape[1]

    avg_corrs_A = np.array([np.nanmean(corr_ests[:,j]) for j in range(d)])
    theoretical_var_reducs_A = avg_corrs_A**2
    avg_corrs_B = np.array([np.corrcoef(vshap_ests_model[:,j], vshap_ests_CV[:,j])[0,1] for j in range(d)])
    theoretical_var_reducs_B = avg_corrs_B**2
    empirical_var_reducs = np.array([1 - (np.nanvar(final_ests[:,j])/np.nanvar(vshap_ests_model[:,j])) for j in range(d)])

    if verbose:
        # print("Using {} method for variance & covariance".format(text))

        print("Empirical variance reductions:\n•", np.round(empirical_var_reducs, 2))
        print("• Average: ", np.round(np.nanmean(empirical_var_reducs), 2))

        print("-"*20, "\nMethod 1: Estimating correlation for each iteration, then averaging")
        print("Average correlations:\n• ", np.round(avg_corrs_A, 3))
        print("Theoretical variance reductions:\n•", np.round(theoretical_var_reducs_A, 2))
        print("• Average: ", np.round(np.nanmean(theoretical_var_reducs_A), 2))

        print("-"*20, "\nMethod 2: Taking correlation of observed Shapley values")
        print("Correlations\n• ", np.round(avg_corrs_B, 3))
        print("Theoretical variance reductions:\n•", np.round(theoretical_var_reducs_B, 2))
        print("• Average: ", np.round(np.nanmean(theoretical_var_reducs_B), 2))

    plt.rcParams["figure.figsize"] = [8, 6]
    # plt.rcParams["figure.autolayout"] = True
    feat_idx = np.arange(d)
    plt.plot(feat_idx, empirical_var_reducs*100, "-o", c="green")
    plt.plot(feat_idx, theoretical_var_reducs_A*100, "-o", c="blue")
    plt.plot(feat_idx, theoretical_var_reducs_B*100, "-o", c="orange")
    plt.legend(['Empirical', 'Theoretical, Average Correlation', 'Theoretical, Correlation between Observations'])
    plt.suptitle("Variance Reduction of CV-adjusted SHAP\n{}".format(message))
    plt.xlabel("Feature index")
    if ylim_zero:
        plt.ylim(0,100)
    plt.yticks(np.arange(11)*10)
    plt.xticks(np.arange(d))
    plt.show()


def rank_shap(shap_vals):
    '''
    Returns indices ranking SHAP values by their absolute value, 
    sorted from highest to lowest.
    '''
    return np.argsort(np.abs(shap_vals))[::-1]

# compute_avg_preds in helper_kshap.py
def map_S(S, mapping_dict):
    '''
    Maps a subset of feature indices to the corresponding subset of columns.
    mapping_dict contains feature indices as keys and their corresponding columns as values.
    '''

    S_cols_list = [mapping_dict[i] for i in S]
    S_cols = sorted([item for sublist in S_cols_list for item in sublist])
    return S_cols

def get_multilevel_features(types_dict):
    '''
    Returns the names of all multilevel features.
    '''
    multilevel_feats = []
    for key, value in types_dict.items():
        if value=="multilevel":
            multilevel_feats.append(key)
    return multilevel_feats

def get_types_dict(X, feature_names, categorical_inds, n_levels):
        '''
        Returns a dictionary whose keys are the feature names and whose values are their type:
        binary, multilevel, continuous, or ordinal. 
        I created this function when we cared a lot about distinguishing between those types;
        now we only care about which features are multilevel. We can implement this more efficiently.
        '''
        d_orig = len(feature_names)
        types = [None]*d_orig
        for j in range(d_orig):
            if j in categorical_inds:
                idx_in_categorical_inds = np.argwhere(np.array(categorical_inds)==j).item()
                n_lev = n_levels[idx_in_categorical_inds]
                if n_lev==2:
                    types[j] = "binary"
                else:
                    types[j] = "multilevel"
            else:
                unique_vals = np.unique(X[:,j]).astype(float)
                vals_minus_int = unique_vals - np.round(unique_vals)
                is_not_ordinal = np.sum(vals_minus_int != 0)
                if is_not_ordinal:
                    types[j] = "continuous"
                else:
                    types[j] = "ordinal"

        types_dict = {}
        for i in range(len(feature_names)):
            types_dict.update({feature_names[i]: types[i]})
        return types_dict

def get_mapping_dict(df_orig, X_df, X_train_raw, categorical_cols):
    '''
    Makes a dictionary mapping features in the original dataset to columns in X.
    We do this if our original dataset has categorical features with more than 2 levels,
    so they need to be represented with multiple binary columns.

    Each key is an index of one of the original features;
    Each value is a list of indices corresponding to that feature in X
    '''
    # make vector listing type of each columnn (ignoring binary structure)
    orig_colnames = df_orig.columns.tolist()[:-1] # NOT in same order as X
    categorical_inds = [orig_colnames.index(col) for col in categorical_cols]
    n_levels = [len(df_orig[col].unique()) for col in categorical_cols]
    types_dict = get_types_dict(X_train_raw, orig_colnames, categorical_inds, n_levels)
    multilevel_colnames = get_multilevel_features(types_dict) # In same order as X

    multilevel_cols_list = []
    for i in range(len(multilevel_colnames)):
        multi_colname = multilevel_colnames[i]
        multilevel_cols_list.append(list(np.where(X_df.columns.str.startswith(multi_colname))[0]))

    X_colnames = list(X_df.columns) # In same order as X

    # Equivalent calculation for d_orig. Uses only multilevel_cols_list
    n_levels_multi = np.sum([len(multilevel_cols_list[i]) for i in range(len(multilevel_cols_list))])
    n_multilevel_feats = len(multilevel_cols_list)
    d_total = len(X_colnames)
    d_true = d_total - n_levels_multi + n_multilevel_feats
    true_colnames_ordered = []
    cur_multi = None
    for i in range(d_total):
        feat_name = X_colnames[i] # In order but too many
        is_multi_vec = [feat_name.startswith(multi_colname) for multi_colname in multilevel_colnames]
        if sum(is_multi_vec)==0: # Feature is not multilevel
            true_colnames_ordered.append(feat_name)
            cur_multi = None
        else:
            multi_name = multilevel_colnames[np.nonzero(is_multi_vec)[0].item()]
            if cur_multi != multi_name:
                true_colnames_ordered.append(multi_name)
                cur_multi = multi_name
            else:
                continue

    mapping_dict = {}
    ct = 0
    multi_idx = 0
    for i in range(d_true):
        feat_name = true_colnames_ordered[i]
        if feat_name not in multilevel_colnames:
            mapping_dict[i] = [ct]
            ct += 1
        else:
            mapping_dict[i] = multilevel_cols_list[multi_idx]
            ct += len(multilevel_cols_list[multi_idx])
            multi_idx += 1
    return mapping_dict

def difference_gradient(model,xloc,sds):
    # calculates a "gradient" based on a 1-standard deviation
    # difference calculation. Here we will use a symmetric
    # difference with 0.5 sd either side. 
    
    grad = []
    for i in range(xloc.shape[1]):
        xlocu, xlocd = np.copy(xloc), np.copy(xloc)
        if sds[i,0]==sds[i,1]:
            xlocu[0,i] = xloc[0,i] + 0.5*sds[i,0]
            xlocd[0,i] = xloc[0,i] - 0.5*sds[i,0]
            grad.append( (model(xlocu)-model(xlocd))/sds[i,0])   
        else:
            xlocu[0,i] = sds[i,1]
            xlocd[0,i] = sds[i,0]
            grad.append( model(xlocu)-model(xlocd))
                 
    return np.array(grad)

def difference_hessian(model,xloc,sds):
    hess = []
    for i in range(xloc.shape[1]):
        xlocu, xlocd = np.copy(xloc), np.copy(xloc)
        xlocu, xlocd = np.copy(xloc), np.copy(xloc)
        if sds[i,0]==sds[i,1]:
            xlocu[0,i] = xloc[0,i] + 0.5*sds[i,0]
            xlocd[0,i] = xloc[0,i] - 0.5*sds[i,0]
            grad = (difference_gradient(model,xlocu,sds) - difference_gradient(model,xlocd,sds))/sds[i,0]
        else:
            xlocu[0,i] = sds[i,1]
            xlocd[0,i] = sds[i,0]
            grad = difference_gradient(model,xlocu,sds) - difference_gradient(model,xlocd,sds)
 
        
        hess.append(grad)
        
    hess = np.array(hess).reshape(xloc.shape[1],xloc.shape[1])
    return (hess + hess.T)/2



