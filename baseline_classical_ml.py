#!/usr/bin/env python
# coding: utf-8

# # Baseline classical ML models

# In[1]:


import numpy as np
from tqdm import tqdm

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, PredefinedSplit, KFold
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB

from sklearn.multioutput import MultiOutputClassifier 


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


fixed_state = 42


# In[4]:


from sklearn.metrics import precision_score, average_precision_score


# In[5]:


A_train = np.load('data/synthetic/A_train_sm.npy', allow_pickle=True)
Y_train = np.load('data/synthetic/Y_train_sm.npy', allow_pickle=True)

A_test = np.load('data/synthetic/A_test_sm.npy', allow_pickle=True)
Y_test = np.load('data/synthetic/Y_test_sm.npy', allow_pickle=True)

A_val = np.load('data/synthetic/A_val_sm.npy', allow_pickle=True)
Y_val = np.load('data/synthetic/Y_val_sm.npy', allow_pickle=True)


# In[6]:


n_classes = 15

fica = FastICA(n_classes + 1, whiten=True)
fica.fit(A_train)
A_train_sc = fica.transform(A_train)
A_val_sc = fica.transform(A_val)
A_test_sc = fica.transform(A_test)


scaler = StandardScaler()
A_train_sc = scaler.fit_transform(np.exp(A_train_sc))
A_val_sc = scaler.transform(np.exp(A_val_sc))
A_test_sc = scaler.transform(np.exp(A_test_sc))


# In[7]:


models_parameters = [
    
    {
        "model" : [MultiOutputClassifier(SVC(kernel="rbf", random_state=fixed_state)) ],
        "model__estimator__kernel" : ["linear", "rbf"],   
        "model__estimator__C" : [0.1, 0.3, 0.5, 1, 10, 50],   
        "model__estimator__degree" : [1, 2, 3, 5],
        "model__estimator__tol" : [1e-1, 1e-2, 1e-3, 1e-4],      
    },
    
    {
        "model" : [MultiOutputClassifier(GaussianNB())],
        "model__estimator__var_smoothing" : [1e-11, 1e-10, 1e-9, 1e-8],   
    },
    
    {
        "model" : [KNeighborsClassifier(n_neighbors=5, n_jobs=-1)],
        "model__n_neighbors" : [3, 5, 10, 15, 20],
        "model__leaf_size" : [10, 20, 30, 40],
    },

    {
        "model" : [RandomForestClassifier(n_estimators=30, max_depth=5, random_state=fixed_state)],
        "model__n_estimators" : [50, 100, 150, 200, 300],   
        "model__max_depth" : [1, 3, 5, 10, 50],  
    },
    
]


# In[8]:


pipeline = Pipeline([
                     ("model", MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)) ),
                     ]) 

for parameters in tqdm(models_parameters): 
    model_fit_params = GridSearchCV(pipeline, parameters, cv=2, scoring="accuracy", verbose=0, n_jobs=-1)
    model_fit_params.fit(A_val_sc, Y_val)
    
    model_fit = model_fit_params.best_estimator_.fit(A_train_sc, Y_train)
    y_pred = model_fit.predict(A_test_sc)
    
    accuracy_cur = f1_score(Y_test, y_pred, average="macro")
    print("For the model {}, score: {}".format(model_fit_params.best_params_, accuracy_cur ))



# # Real data

# In[9]:


A_train = np.load('data/synthetic/A_train_sm_real.npy', allow_pickle=True)
Y_train = np.load('data/synthetic/Y_train_sm_real.npy', allow_pickle=True)

A_test = np.load('data/synthetic/A_test_sm_real.npy', allow_pickle=True)
Y_test = np.load('data/synthetic/Y_test_sm_real.npy', allow_pickle=True)

A_val = np.load('data/synthetic/A_val_sm_real.npy', allow_pickle=True)
Y_val = np.load('data/synthetic/Y_val_sm_real.npy', allow_pickle=True)


# In[10]:


n_classes = 15

fica = FastICA(n_classes + 1, whiten=True)
fica.fit(A_train)
A_train_sc = fica.transform(A_train)
A_val_sc = fica.transform(A_val)
A_test_sc = fica.transform(A_test)


scaler = StandardScaler()
A_train_sc = scaler.fit_transform(np.exp(A_train_sc))
A_val_sc = scaler.transform(np.exp(A_val_sc))
A_test_sc = scaler.transform(np.exp(A_test_sc))


# In[11]:


pipeline = Pipeline([
                     ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
                     ]) 

for parameters in tqdm(models_parameters): 
    model_fit_params = GridSearchCV(pipeline, parameters, cv=2, scoring="accuracy", verbose=0, n_jobs=-1)
    model_fit_params.fit(A_val_sc, Y_val)
    
    model_fit = model_fit_params.best_estimator_.fit(A_train_sc, Y_train)
    y_pred = model_fit.predict(A_test_sc)
    
    accuracy_cur = f1_score(Y_test, y_pred, average="macro")
    print(accuracy_cur)


