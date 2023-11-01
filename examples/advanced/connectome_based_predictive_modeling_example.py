"""
Connectome-based predictive modeling

CPM is a method described in the following Nature Protocols article: https://www.nature.com/articles/nprot.2016.178
It has been used in a number of publications to predict behavior from connectivity data.
CPM works similar to a feature selection method. First, relevant edges (connectivity values) are identified through
correlation analysis. Every edge is correlated with the predictive target. Only significant edges will be used in the
subsequent steps. Next, the edge values for all significant positive and for all significant negative correlations are
summed to create two new features. Lastly, these two features are used as input to another classifier.

In this example, no connectivity data is used, but the method will still work.
This example is just supposed to show how to use CPM as feature selection and integration tool in PHOTONAI.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai import Hyperpipe, PipelineElement


X, y = load_breast_cancer(return_X_y=True)

pipe = Hyperpipe("cpm_feature_selection_pipe",
                  outer_cv=KFold(n_splits=5, shuffle=True, random_state=15),
                  inner_cv=KFold(n_splits=5, shuffle=True, random_state=15),
                  metrics=["balanced_accuracy"], best_config_metric="balanced_accuracy",
                  project_folder='./tmp')

pipe += PipelineElement('CPMFeatureSelection', hyperparameters={'corr_method': ['pearson', 'spearman'],
                                                                'p_threshold': [0.01, 0.05]})

pipe += PipelineElement('LogisticRegression')

pipe.fit(X, y)