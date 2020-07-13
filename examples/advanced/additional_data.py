import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin

from photonai.base import Hyperpipe, PipelineElement, OutputSettings


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


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)

my_pipe.add(PipelineElement('StandardScaler'))


my_pipe += PipelineElement.create("CustomWrapper", AdditionalDataWrapper(), hyperparameters={})

my_pipe.fit(X, y, true_predictions=np.array(y))

