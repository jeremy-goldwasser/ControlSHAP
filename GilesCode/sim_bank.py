


import numpy as np
from helper import *
from helper2_dep import *
from helper2_indep import *
from helper2_shapley_sampling import *
from helper3_kshap import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
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


# # Train neural net

# ### Define network class & instance

# In[21]:


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

class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.softmax(out)
        return out


# ### Turn dataset into Pytorch iterable

# In[23]:


# Convert the input and label data to PyTorch tensors
inputs = torch.tensor(X_train, dtype=torch.float32)
labels = torch.tensor(Y_train, dtype=torch.long)

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


# ### Train network. 

# In[25]:


torch.manual_seed(0)

# Create an instance
net = TwoLayerNet(input_size=full_dim, hidden_size=50, output_size=2)
# net = ThreeLayerNet(input_size=full_dim, hidden_size1=50, hidden_size2=50, output_size=2)
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


# ### Compute training outputs and errors
# - Looks good!

# In[33]:


test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs = net(test_tensor)

n_test = X_test.shape[0]
n_positive_preds = torch.sum(np.argmax(outputs, axis=1)==1).item()
print("{}/{} predictions are for positive class; really {}"
    .format(n_positive_preds,n_test, np.sum(Y_test==1)))
Y_preds = torch.argmax(outputs, axis=1)
print("Balanced sampling. {}% accuracy".format(round(100*(np.sum(Y_test==Y_preds.numpy())/n_test))))


# ## Compute Gradient and Hessian of neural net w.r.t. input

# In[34]:


xloc_torch = torch.tensor(xloc, dtype=torch.float32).requires_grad_(True)
y_pred = net(xloc_torch)[0,1]
y_pred.backward()
gradient = xloc_torch.grad.detach().numpy().reshape((full_dim, 1))


# In[35]:


def neural_net(x):
    output = net(x)[0,1] if x.shape[0]==1 else net(x)[:,1]
    return output

def compute_hessian(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    hessian = torch.autograd.functional.hessian(neural_net, x)
    hessian = hessian.reshape((full_dim,full_dim)).detach().numpy()
    return hessian

hessian = compute_hessian(xloc)
def f_model(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return neural_net(x).detach().numpy()


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
    xloc_torch = torch.tensor(xloci, dtype=torch.float32).requires_grad_(True)
    y_pred = net(xloc_torch)[0,1]
    y_pred.backward()
    gradient = xloc_torch.grad.detach().numpy().reshape((full_dim, 1))
    hessian = compute_hessian(xloci)
    
    shap_CV_true_indep = compute_true_shap_cv_indep(xloci, gradient, hessian, feature_means, cov_mat, mapping_dict)
    shap_CV_true_dep = linear_shap_vals(xloci, D_matrices, feature_means, gradient,mapping_dict=mapping_dict)
    
    sims_kshap_indep = []
    sims_kshap_dep = []
    sims_ss_indep = []
    sims_ss_dep = []
    for j in range(nsim_per_point):
        print([i,j])
        independent_features=True
        obj_kshap_indep = cv_kshap_compare(f_model, X_train, xloci,
                            independent_features,
                            gradient, hessian,
                            shap_CV_true=shap_CV_true_indep,
                            M=1000, n_samples_per_perm=10, K = 100, n_boot=250,
                            mapping_dict=mapping_dict)        
        sims_kshap_indep.append(obj_kshap_indep)
        
        obj_ss_indep = cv_shapley_sampling(f_model, X_train, xloci, 
                                independent_features,
                                gradient, hessian, shap_CV_true=shap_CV_true_indep,
                                M=1000, n_samples_per_perm=10,
                                mapping_dict=mapping_dict)
        sims_ss_indep.append(obj_ss_indep)
        
        independent_features=False
        obj_kshap_dep = cv_kshap_compare(f_model, X_train, xloci,
                            independent_features,
                            gradient,
                            shap_CV_true=shap_CV_true_dep,
                            M=1000, n_samples_per_perm=10, cov_mat=cov2, 
                            K = 100, n_boot=250,
                            mapping_dict=mapping_dict)
        sims_kshap_dep.append(obj_kshap_dep)
        
        obj_ss_dep = cv_shapley_sampling(f_model, X_train, xloci, 
                                independent_features,
                                gradient, shap_CV_true=shap_CV_true_dep,
                                M=1000, n_samples_per_perm=10, cov_mat=cov2,
                                mapping_dict=mapping_dict)
        sims_ss_dep.append(obj_ss_dep)

    kshaps_indep.append(sims_kshap_indep)
    kshaps_dep.append(sims_kshap_dep)
    sss_indep.append(sims_ss_indep)
    sss_dep.append(sims_ss_dep)

    np.save('bank_kshap_indep.npy',kshaps_indep)
    np.save('bank_kshap_dep.npy',kshaps_dep)
    np.save('bank_ss_indep.npy',sss_indep)
    np.save('bank_ss_dep.npy',sss_dep)