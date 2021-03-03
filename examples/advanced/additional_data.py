import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from photonai.base import Hyperpipe, PipelineElement


class AdditionalDataWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        if "true_predictions" in kwargs:
            print("Found additional data")
        return self

    def predict(self, X, **kwargs):
        y_true = kwargs["true_predictions"]
        assert X.shape[0] == len(y_true)
        return y_true

    def save(self):
        return None


my_pipe = Hyperpipe('additional_data_pipe',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement.create("CustomWrapper", AdditionalDataWrapper(), hyperparameters={})

X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y, true_predictions=np.array(y))
