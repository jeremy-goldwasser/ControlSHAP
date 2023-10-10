import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
import pickle
from os.path import join
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import lightgbm as lgb


from helper2 import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper4_kshap import *

def fullsim(fname,X_locs,X,model,gradfn,hessfn,cov_mat,D_matrices,mapping_dict=[],sds=[],
            nsim_per_point=50,M=1000,n_samples_per_perm=[10,1],K=50, n_boot=100):
    
    n_pts = X_locs.shape[0]
    d = X.shape[1]
    
    feature_means = np.mean(X, axis=0)
    
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
                                    M=M, n_samples_per_perm=n_samples_per_perm[0], K = K, n_boot=n_boot,
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
                                        M=M, n_samples_per_perm=n_samples_per_perm[1],
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
                                    M=M, n_samples_per_perm=n_samples_per_perm[0], K = K, n_boot=n_boot, 
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
                                        M=M, n_samples_per_perm=n_samples_per_perm[1], cov_mat=cov_mat,
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
        
        
def correct_cov(cov_mat,Kr):
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
        
    return (cov2+cov2.T)/2


def loaddata(name):
    if name == 'logistic':
        d = 10
        FEATURE_MEANS = np.repeat(0, d)
        FEATURE_VARS = np.repeat(1, d)
        FEATURE_COVS = [0.5, 0.25]
        COV_MAT = make_banded_cov(FEATURE_VARS, FEATURE_COVS)

        # Randomly generate samples
        np.random.seed(1)
        X = np.random.multivariate_normal(FEATURE_MEANS, COV_MAT, size=10000)
        X_train, X_test = X[:8000], X[8000:]
        xloc = X_test[10].reshape((1,d))

        np.random.seed(1)
        BETA = np.random.normal(0, 1, size = d)
        def model(x):
            yhat = sigmoid(np.dot(x, BETA))
            return yhat.item() if x.shape[0]==1 else yhat
        y = (model(X) > 0.5).astype(int)
        y_train, y_test = y[:8000], y[8000:]
        mapping_dict = None
    elif name == 'bank':
        dirpath = "../Data/bank"
        # dirpath = /PATH/TO/DATA
        df_orig = pd.read_csv(join(dirpath, "df_orig.csv"))

        X_train_raw = np.load(join(dirpath, "X_train.npy"))
        X_test_raw = np.load(join(dirpath, "X_test.npy"))
        y_train = np.load(join(dirpath, "Y_train.npy"))
        y_test = np.load(join(dirpath, "Y_test.npy"))
        full_dim = X_train_raw.shape[1] # dimension including all binarized categorical columns
        X_df = pd.read_csv(join(dirpath, "X_df.csv"))


        trainmean, trainstd = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
        def rescale(x, trainmean, trainstd):
            return (x - trainmean) / trainstd
        X_train = rescale(X_train_raw, trainmean, trainstd)
        X_test = rescale(X_test_raw, trainmean, trainstd)

        feature_means = np.mean(X_train, axis=0)
        cov_mat = np.cov(X_train, rowvar=False)


        df_orig.columns = df_orig.columns.str.replace(' ', '_')
        categorical_cols = ['Job', 'Marital', 'Education', 'Default', 'Housing',
                            'Loan', 'Contact', 'Month', 'Prev_Outcome']


        mapping_dict = get_mapping_dict(df_orig, X_df, X_train_raw, categorical_cols)

        
    elif name == "brca":
        data = pd.read_csv('../Data/brca_small.csv')
        X = data.values[:, :-1][:,:20]
        Y = data.values[:, -1]
        Y = (Y==2).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=100, random_state=0)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=100, random_state=1)

        # Normalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        # X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        mapping_dict = None
        
    elif name == "census":
        X_train = np.load('../Data/census/X_train.npy')
        y_train = np.load('../Data/census/y_train.npy')

        X_test = np.load('../Data/census/X_test.npy')
        y_test = np.load('../Data/census/y_test.npy')

        mapping_dict = pickle.load(open('../Data/census/censusmapping.p','rb'))

    elif name == "credit":
        X_train = np.load('../Data/credit/X_train.npy')
        y_train = np.load('../Data/credit/y_train.npy')

        X_test = np.load('../Data/credit/X_test.npy')
        y_test = np.load('../Data/credit/y_test.npy')

        mapping_dict = pickle.load(open('../Data/credit/creditmapping.p','rb'))      

    else: print("Data unrecognized")
        
    return X_train, y_train, X_test, y_test, mapping_dict



def fitmodgradhess(mod,X_train,y_train,X_test,y_test):
    if mod == "rf":
        rf = RandomForestClassifier().fit(X_train, y_train)
        print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
        print("Estimation accuracy: {}".format(np.mean((rf.predict(X_test) > 0.5)==y_test)*100))


        def model(xloc):
            return rf.predict_proba(xloc)[:,1]


        def gradfn(model,xloc,sds):
            return difference_gradient(model,xloc,sds)


        def hessfn(model,xloc,sds):
            return difference_hessian(model,xloc,sds)


        d = X_train.shape[1]
        feature_means = np.mean(X_train, axis=0)
        xloc = X_test[0].reshape((1,d))
        cov_mat = np.cov(X_train, rowvar=False)            
        cov_mat = correct_cov(cov_mat,Kr=10000)

        sds = []
        for i in range(d):
            uu = np.unique(X_train[:,i])
            if len(uu) == 2:
                sds.append(uu)
            else:
                mi = np.delete(np.arange(d),i)
                cm,ccov = compute_cond_mean_and_cov(xloc,mi,feature_means,cov_mat)
                sds.append(np.repeat(np.sqrt(ccov),2))
        
                #sds.append(np.repeat(np.std(X_train[:,i]),2))
        sds = np.array(sds)
    
    elif mod == "glm":
        logreg = LogisticRegression().fit(X_train, y_train)
        print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
        print("Estimation accuracy: {}".format(np.mean((logreg.predict(X_test) > 0.5)==y_test)*100))

        def model(xloc):
            return logreg.predict_proba(xloc)[:,1]

        BETA = logreg.coef_.reshape(-1)

        def gradfn(model,xloc,BETA):
            return logreg_gradient(model, xloc, BETA)


        def hessfn(model,xloc,BETA):
            return logreg_hessian(model, xloc, BETA)

        sds = BETA
    
    elif mod == "gbm":
        d = X_train.shape[1]
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
        print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
        print("Estimation accuracy: {}".format(np.mean((lgbmodel.predict(X_test) > 0.5)==y_test)*100))


        def model(xloc):
            return lgbmodel.predict(xloc)


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
        
    elif mod == "nn":
        d = X_test.shape[1]
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

        class TwoLayerNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(TwoLayerNet, self).__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.tanh = nn.Tanh()
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                out = self.linear1(x)
                out = self.tanh(out)
                out = self.linear2(out)
                out = self.softmax(out)
                return out

        # Convert the input and label data to PyTorch tensors
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)

        # Compute the class weights
        class_counts = torch.bincount(labels)
        num_samples = len(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]

        # Create a sampler with balanced weights
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        # Create a DataLoader with the sampler
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)


        torch.manual_seed(0)

        # Create an instance
        net = TwoLayerNet(input_size=d, hidden_size=50, output_size=2)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()#weight=torch.tensor(weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#.01

        # Iterate over the training data in batches
        num_epochs = 5

        # Train the network for the specified number of epochs
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                # Zero the gradients for this batch
                optimizer.zero_grad()

                # Compute the forward pass of the network
                outputs = net(inputs)

                # Compute the loss for this batch
                loss = criterion(outputs, labels)

                # Compute the gradients of the loss with respect to the parameters
                loss.backward()

                # Update the parameters using the optimizer
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")



        test_tensor = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            outputs = net(test_tensor)

        n_test = X_test.shape[0]
        n_positive_preds = torch.sum(np.argmax(outputs, axis=1)==1).item()
        print("{}/{} predictions are for positive class; really {}"
            .format(n_positive_preds,n_test, np.sum(y_test==1)))
        Y_preds = torch.argmax(outputs, axis=1)
        print("Balanced sampling. {}% accuracy".format(round(100*(np.sum(y_test==Y_preds.numpy())/n_test))))


        def neural_net(x):
            output = net(x)[0,1] if x.shape[0]==1 else net(x)[:,1]
            return output

        def model(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            return neural_net(x).detach().numpy()

        def gradfn(model,x,sds):
            xloc_torch = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
            y_pred = net(xloc_torch)[0,1]
            y_pred.backward()
            gradient = xloc_torch.grad.detach().numpy().reshape((d, 1))
            return gradient


        def hessfn(model,x,sds):
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            hessian = torch.autograd.functional.hessian(neural_net, x)
            hessian = hessian.reshape((d,d)).detach().numpy()
            return hessian

        sds = None
    
    else: print("Model not known")

    return model, gradfn, hessfn, sds
