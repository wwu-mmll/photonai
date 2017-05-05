
# coding: utf-8


import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch
from sklearn.model_selection import KFold


# ## Test HyperPipe using the BreastCancer dataset


# classification of malignant vs. benign tumors
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# create cross-validation object first
cv_object = KFold(n_splits=3, shuffle=True, random_state=0)

# create a hyperPipe
manager = Hyperpipe('god', cv_object, optimizer='random_grid_search')

pca_preproc = PipelineElement.create('pca', {'n_components': [1, None]}, set_disabled=True)
scaler_preproc = PipelineElement.create('standard_scaler', {}, set_disabled=True)

# SVMs (linear and rbf)
svc_estimator = PipelineElement.create('svc', {}, kernel='linear')
# Logistic Regression with different C values (5 sets)
lr_estimator = PipelineElement.create('logistic', {'C': np.logspace(-4, 4, 5)})

# use either standard scaler or PCA
preproc_obj = PipelineSwitch('pre', [pca_preproc, scaler_preproc])
manager.add(preproc_obj)

# and then user either SVC or LR
est = PipelineSwitch('final_estimator', [svc_estimator, lr_estimator])
manager.add(est)
manager.fit(X, y)






