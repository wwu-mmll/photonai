
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
print(X.shape)
print(y.shape)

# create cross-validation object first
cv_object = KFold(n_splits=3, shuffle=True, random_state=0)


# # now do the same with hyperPipe

# create a hyperPipe
manager = Hyperpipe('god', cv_object, optimizer='random_grid_search')
# manager = Hyperpipe('god', cv_object, optimizer='random_grid_search', optimizer_params={'k': 4})
# manager = Hyperpipe('god', cv_object, optimizer='timeboxed_random_grid_search',
#                     optimizer_params={'limit_in_minutes': 1})

pca_preproc = PipelineElement.create('pca', {'n_components': [1, None]}, set_disabled=True)
scaler_preproc = PipelineElement.create('standard_scaler', {}, set_disabled=True)

# SVMs (linear and rbf)
svc_estimator = PipelineElement.create('svc', {}) #'kernel': ['linear']
# Logistic Regression with different C values (5 sets)
lr_estimator = PipelineElement.create('logistic', {'C': np.logspace(-4, 4, 5)})

# ... trotzdem sind jetzt nur zwei Elemente drin
preproc_obj = PipelineSwitch('pre', [pca_preproc, scaler_preproc])
manager.add(preproc_obj)

# bei den estimators klappt es dann komischer Weise...
est = PipelineSwitch('final_estimator', [svc_estimator, lr_estimator])
manager.add(est)

# sieht irgendwie aus wie 6 folds (obwohl oben 3 eingestellt ist)
manager.fit(X, y)
print(len(manager.performance_history))





