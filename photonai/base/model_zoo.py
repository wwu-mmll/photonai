import logging
from photonai.base.hyperpipe import Hyperpipe, OutputSettings
from photonai.base.photon_elements import PipelineElement, Switch, Stack
from photonai.optimization.hyperparameters import IntegerRange
from photonai.optimization import FloatRange
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits
from photonai.photonlogger.logger import logger
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
import pandas as pd


class ClassifierSwitch(Switch, ClassifierMixin):

    def __init__(self, name: str = 'classifier_switch') -> None:
        element_list = [PipelineElement('SVC', hyperparameters={'C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8],
                                                                'kernel': ['rbf', 'linear']}, max_iter=1000),
                        PipelineElement('RandomForestClassifier', hyperparameters={"max_features": ["sqrt", "log2"],
                                                                                   "min_samples_leaf": [0.01, 0.1, 0.2]}),
                        PipelineElement('AdaBoostClassifier', hyperparameters={'n_estimators': [10, 25, 50]}),
                        PipelineElement('LogisticRegression',
                                        hyperparameters={"C": [1e-4, 1e-2, 1, 1e2, 1e4],
                                                         "penalty": ['l1', 'l2']},
                                        solver='saga', n_jobs=1),
                        PipelineElement('GaussianNB'),
                        PipelineElement('KNeighborsClassifier', hyperparameters={"n_neighbors": [5, 10, 15]})]
        # todo: make a dedicated documentation site
        super(ClassifierSwitch, self).__init__(name, elements=element_list)


class RegressorSwitch(Switch, RegressorMixin):

    def __init__(self, name: str = 'regressor_switch') -> None:
        element_list = [PipelineElement('SVR', hyperparameters={'C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8],
                                                                'kernel': ['rbf', 'linear']}, max_iter=1000),
                        PipelineElement('RandomForestRegressor', hyperparameters={"max_features": ["sqrt", "log2"],
                                                                                  "min_samples_leaf": [0.01, 0.1,
                                                                                                       0.2]}),
                        PipelineElement('AdaBoostRegressor', hyperparameters={'n_estimators': [10, 25, 50]}),
                        PipelineElement('LinearRegression'),
                        PipelineElement('KNeighborsRegressor', hyperparameters={"n_neighbors": [5, 10, 15]})]

        super(RegressorSwitch, self).__init__(name, elements=element_list)


class DefaultPipeline(Hyperpipe):

    def __init__(self,
                 name: str = 'default_pipeline',
                 project_folder: Union[str, Path] = './',
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=10),
                 outer_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=5),
                 metrics: list = None,
                 best_config_metric: str = '',
                 optimizer: str = 'random_grid_search',
                 optimizer_params: dict = {'n_configurations': 25},
                 X_csv_path: Union[str, Path, None] = None,
                 y_csv_path: Union[str, Path, None] = None,
                 delimiter=",",
                 add_default_pipeline_elements: bool = True,
                 scaling: bool = True,
                 imputation: bool = False,
                 imputation_nan_value=np.nan,
                 feature_selection: bool = False,
                 dim_reduction: bool = False,
                 n_pca_components: Union[int, float, None, IntegerRange, FloatRange] = None,
                 add_estimator: bool = True,
                 default_estimator: BaseEstimator = None,
                 **kwargs):

        self.X_csv_path = X_csv_path
        self.y_csv_path = y_csv_path
        self.delimiter = delimiter

        super(DefaultPipeline, self).__init__(name=name,
                                              project_folder=project_folder,
                                              inner_cv=inner_cv,
                                              outer_cv=outer_cv,
                                              metrics=metrics,
                                              best_config_metric=best_config_metric,
                                              optimizer=optimizer,
                                              optimizer_params=optimizer_params,
                                              **kwargs)

        if add_default_pipeline_elements is True:
            self.set_default_pipeline(scaling, imputation, imputation_nan_value, feature_selection, dim_reduction,
                                      n_pca_components, add_estimator, default_estimator)

    def set_default_pipeline(self, scaling, imputation, imputation_nan_value, feature_selection, dim_reduction,
                             n_pca_components, add_estimator, default_estimator):
        logger.stars()
        if scaling is True:
            logger.photon_system_log("USING STANDARD SCALER ")
            self.add(PipelineElement('StandardScaler'))
        if imputation is True:
            logger.photon_system_log("USING SIMPLE IMPUTER ")
            self.add(PipelineElement('SimpleImputer', missing_values=imputation_nan_value))
        if feature_selection:
            self.add(PipelineElement('FClassifSelectPercentile',
                                     hyperparameters={'percentile': IntegerRange(10, 50, step=10)},
                                     test_disabled=True))
        if dim_reduction is True:
            logger.photon_system_log("USING PCA ")
            self.add(PipelineElement('PCA', n_components=n_pca_components, test_disabled=True))
        if add_estimator is True:
            if default_estimator is None:
                raise ValueError("If paramater add_estimator is True, then default_estimator cannot be None. Use a "
                                 "string keyword argument such as 'SVC' or 'SVR' or photon native constructs "
                                 "such as PipelineElements, Switchs oder Stacks")
            ve = ValueError("Currently only string keywords such as 'SVC' or 'SVR' or photon native constructs "
                                 "such as PipelineElements, Switchs oder Stacks are allowed")
            if isinstance(default_estimator, str):
                estimator = PipelineElement(default_estimator)
            else:
                try:
                    estimator = default_estimator()
                    if not isinstance(estimator, (PipelineElement, Switch, Stack, ClassifierSwitch, RegressorSwitch)):
                        raise ve
                except Exception as e:
                    logging.error(e)
                    raise ve
            self += estimator
            logger.photon_system_log("USING " + self.elements[-1].name.upper())
            if hasattr(self.elements[-1], 'elements'):
                for idx, e in enumerate(self.elements[-1].elements):
                    logger.photon_system_log(e.name)
                    logger.photon_system_log(e.initial_hyperparameters)
                    logger.photon_system_log("---")
        logger.stars()

    def fit(self, X=None, y=None):
        if (X is not None and self.X_csv_path is not None) or (y is not None and self.y_csv_path is not None):
            raise ValueError("You can either give the fit function data or the pipe definition paths "
                             "to csv files to load data from. Not both.")

        if (X is None and self.X_csv_path is None) or (y is None and self.y_csv_path is None):
            raise ValueError("Please give either the fit function array-like X and y data "
                             "or give the pipe definition paths to csv files to load X and y data from.")

        X = X if X is not None else pd.read_csv(self.X_csv_path, delimiter=self.delimiter)
        y = y if y is not None else pd.read_csv(self.y_csv_path, delimiter=self.delimiter)
        super().fit(X, y)


class ClassificationPipe(DefaultPipeline):

    def __init__(self,
                 name: Optional[str] = 'classification_pipeline',
                 project_folder: Union[str, Path] = './classification/',
                 metrics: list = None,
                 best_config_metric: str = 'balanced_accuracy',
                 default_estimator: Union[BaseEstimator, ClassifierSwitch] = ClassifierSwitch,
                 **kwargs):

        metrics = metrics if metrics is not None else ['balanced_accuracy',
                                                       'specificity',
                                                       'precision',
                                                       'recall',
                                                       'f1_score']

        super(ClassificationPipe, self).__init__(name=name,
                                                 project_folder=project_folder,
                                                 metrics=metrics,
                                                 best_config_metric=best_config_metric,
                                                 default_estimator=default_estimator,
                                                 **kwargs)


class RegressionPipe(DefaultPipeline):

    def __init__(self,
                 name: Optional[str] = 'classification_pipeline',
                 project_folder: Union[str, Path] = './classification/',
                 metrics: list = None,
                 best_config_metric: str = 'mean_squared_error',
                 default_estimator: Union[BaseEstimator, RegressorSwitch] = RegressorSwitch,
                 **kwargs):

        metrics = metrics if metrics is not None else ['mean_absolute_error',
                                                       'mean_squared_error',
                                                       'explained_variance']

        super(RegressionPipe, self).__init__(name=name,
                                             project_folder=project_folder,
                                             metrics=metrics,
                                             best_config_metric=best_config_metric,
                                             default_estimator=default_estimator,
                                             **kwargs)
