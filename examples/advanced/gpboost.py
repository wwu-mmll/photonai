# pip install gpboost -U
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold, KFold
from photonai.base import Hyperpipe, PipelineElement
import numpy as np
import pandas as pd
import gpboost as gpb
# from gpboost import GPBoostRegressor


class GPBoostDataWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.needs_covariates = True
        # self.gpmodel = gpb.GPModel(likelihood="gaussian")
        self.gpboost = None


    def fit(self, X, y, **kwargs):
        self.gpboost = gpb.GPBoostRegressor()
        if "clusters" in kwargs:
            clst = pd.Series(kwargs["clusters"])
            gpmodel = gpb.GPModel(likelihood="gaussian", group_data=clst)
            self.gpboost.fit(X, y, gp_model=gpmodel)
        else:
            raise NotImplementedError("GPBoost needs clusters")
        return self

    def predict(self, X, **kwargs):
        clst = pd.Series(kwargs["clusters"])
        preds = self.gpboost.predict(X, group_data_pred=clst)
        preds = preds["response_mean"]
        return preds

    def save(self):
        return None


def get_gpboost_pipe(pipe_name, project_folder, split="group"):

    if split == "group":
        outercv = GroupKFold(n_splits=10)
    else:
        outercv = KFold(n_splits=10)

    my_pipe = Hyperpipe(pipe_name,
                        optimizer='grid_search',
                        metrics=['mean_absolute_error', 'mean_squared_error',
                                 'spearman_correlation', 'pearson_correlation'],
                        best_config_metric='mean_absolute_error',
                        outer_cv=outercv,
                        inner_cv=KFold(n_splits=10),
                        calculate_metrics_across_folds=True,
                        use_test_set=True,
                        verbosity=1,
                        project_folder=project_folder)

    # Add transformer elements
    my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                               test_disabled=True, with_mean=True, with_std=True)

    my_pipe += PipelineElement.create("GPBoost", GPBoostDataWrapper(), hyperparameters={})

    return my_pipe


def get_mock_data():

    X = np.random.randint(10, size=(200, 9))
    y = np.sum(X, axis=1)
    clst = np.random.randint(10, size=200)

    return X, y, clst


if __name__ == '__main__':


    X, y, clst = get_mock_data()

    # define project folder
    project_folder = "./tmp/gpboost_debug"

    my_pipe = get_gpboost_pipe("Test_gpboost", project_folder, split="random")
    my_pipe.fit(X, y, clusters=clst)
