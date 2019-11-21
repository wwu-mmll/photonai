import os
import pickle
import numpy as np
from photonai.processing.metrics import Scorer
from photonai.helper.helper import PhotonDataHelper


class GroupVsAllAnalysis:

    def __init__(self, analysis_folder, hyperpipe_ctor, group_var):
        self.analysis_folder = analysis_folder
        self.hyperpipe_ctor = hyperpipe_ctor
        self.group_var = group_var
        self.metrics = None

    def fit(self, X, y, **kwargs):
        # whole dataset
        whole_hyperpipe_predictions = self.fit_hyperpipe(X, y, **kwargs)
        self.process_testset_predictions("entire_dataset", whole_hyperpipe_predictions)

        # group-wise
        groups = np.unique(self.group_var)
        split_dict = {}
        for group in groups:
            group_indices = np.where(y == group)
            split_dict[group] = dict()
            split_dict[group]["indices"] = group_indices
            group_X, group_y, group_kwargs = PhotonDataHelper.split_data(X, y, kwargs, indices=group_indices)
            group_predictions = self.fit_hyperpipe(group_X, group_y, **group_kwargs)
            split_dict[group]["y_pred"] = group_predictions["y_pred"]
            split_dict[group]["y_true"] = group_predictions["y_true"]
            self.process_testset_predictions(str(group), group_predictions)

        # whole dataset group-wise
        # collect all indices, then collect y_true and y_pred and sort back to original sequence
        group_prediction_indices = np.array(list().extend([split_dict[group]["indices"] for group in groups]))
        group_prediction_y_true = np.array(list().extend([split_dict[group]["y_true"] for group in groups]))[group_prediction_indices]
        group_prediction_y_pred = list().extend([split_dict[group]["y_pred"] for group in groups])[group_prediction_indices]
        self.process_testset_predictions("groupwise_dataset", {'y_true': group_prediction_y_true, 'y_pred': group_prediction_y_pred})

    def fit_hyperpipe(self, X, y, **kwargs):
        hyperpipe = self.hyperpipe_ctor()
        if self.metrics is None:
            self.metrics = hyperpipe.optimization.metrics()
        hyperpipe.fit(X, y, **kwargs)
        prediction_dict = hyperpipe.results_handler.get_test_predictions()
        return prediction_dict

    def process_testset_predictions(self, name, prediction_dict):
        total_testset_metrics = self.get_metrics(prediction_dict)
        prediction_dict["metrics"] = total_testset_metrics
        prediction_dict["name"] = name
        pickle.dump(prediction_dict, os.path.join(self.analysis_folder, name + ".p"))

    def get_metrics(self, test_predictions_dict):
        return Scorer.calculate_metrics(test_predictions_dict["y_true"], test_predictions_dict["y_pred"], self.metrics)


