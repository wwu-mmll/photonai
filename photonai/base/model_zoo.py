from photonai.base.hyperpipe import Hyperpipe, OutputSettings
from photonai.base.photon_elements import PipelineElement, Switch
from photonai.optimization import FloatRange
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits
from photonai.photonlogger.logger import logger
from typing import Optional, List, Union
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


class DefaultPipelineGenerator:

    def set_default_pipeline(self,
                             scaling: bool = True,
                             imputation: bool = False,
                             imputation_nan_value=np.nan,
                             dim_reduction: bool = False,
                             n_pca_components: Union[int, float, None] = None,
                             add_estimator: bool = True):
        logger.stars()
        if scaling is True:
            logger.photon_system_log("USING STANDARD SCALER ")
            self.add(PipelineElement('StandardScaler'))
        if imputation is True:
            logger.photon_system_log("USING SIMPLE IMPUTER ")
            self.add(PipelineElement('SimpleImputer', missing_values=imputation_nan_value))
        if dim_reduction is True:
            logger.photon_system_log("USING PCA ")
            self.add(PipelineElement('PCA', n_components=n_pca_components))
        if add_estimator is True:
            if not hasattr(self, 'default_estimator_cls'):
                raise ValueError("Must have estimator_cls attribute set to an estimator class")

            self += self.default_estimator_cls()
            logger.photon_system_log("USING " + self.elements[-1].name.upper())
            for idx, e in enumerate(self.elements[-1].elements):
                logger.photon_system_log(e.name)
                logger.photon_system_log(e.initial_hyperparameters)
                logger.photon_system_log("---")
        logger.stars()


class ClassificationPipe(Hyperpipe, DefaultPipelineGenerator):

    def __init__(self,
                 name: Optional[str],
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=10),
                 outer_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=5),
                 optimizer: str = 'grid_search',
                 optimizer_params: dict = {},
                 metrics: list = ['balanced_accuracy', 'specificity', 'precision', 'recall', 'f1_score'],
                 best_config_metric: str = 'balanced_accuracy',
                 eval_final_performance: bool = None,
                 use_test_set: bool = True,
                 test_size: float = 0.2,
                 project_folder: str = '',
                 calculate_metrics_per_fold: bool = True,
                 calculate_metrics_across_folds: bool = False,
                 ignore_sanity_checks: bool = False,
                 random_seed: int = None,
                 verbosity: int = 0,
                 learning_curves: bool = False,
                 learning_curves_cut: FloatRange = None,
                 output_settings: OutputSettings = None,
                 performance_constraints: list = None,
                 permutation_id: str = None,
                 cache_folder: str = None,
                 nr_of_processes: int = 1,
                 multi_threading: bool = True,
                 allow_multidim_targets: bool = False,
                 default_estimator_cls: Union[BaseEstimator, ClassifierSwitch] = ClassifierSwitch):

        self.default_estimator_cls = default_estimator_cls

        super(ClassificationPipe, self).__init__(name, inner_cv, outer_cv, optimizer, optimizer_params,
                                                 metrics, best_config_metric,
                                                 eval_final_performance, use_test_set, test_size,
                                                 project_folder, calculate_metrics_per_fold,
                                                 calculate_metrics_across_folds,
                                                 ignore_sanity_checks, random_seed, verbosity, learning_curves,
                                                 learning_curves_cut, output_settings, performance_constraints,
                                                 permutation_id, cache_folder, nr_of_processes, multi_threading,
                                                 allow_multidim_targets)


class RegressionPipe(Hyperpipe, DefaultPipelineGenerator):

    def __init__(self,
                 name: Optional[str],
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=10),
                 outer_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=5),
                 optimizer: str = 'grid_search',
                 optimizer_params: dict = {},
                 metrics: list = ['mean_absolute_error', 'mean_squared_error'],
                 best_config_metric: str = 'mean_squared_error',
                 eval_final_performance: bool = None,
                 use_test_set: bool = True,
                 test_size: float = 0.2,
                 project_folder: str = '',
                 calculate_metrics_per_fold: bool = True,
                 calculate_metrics_across_folds: bool = False,
                 ignore_sanity_checks: bool = False,
                 random_seed: int = None,
                 verbosity: int = 0,
                 learning_curves: bool = False,
                 learning_curves_cut: FloatRange = None,
                 output_settings: OutputSettings = None,
                 performance_constraints: list = None,
                 permutation_id: str = None,
                 cache_folder: str = None,
                 nr_of_processes: int = 1,
                 multi_threading: bool = True,
                 allow_multidim_targets: bool = False,
                 default_estimator_cls: Union[BaseEstimator, RegressorSwitch] = RegressorSwitch):

        self.default_estimator_cls = default_estimator_cls

        super(RegressionPipe, self).__init__(name, inner_cv, outer_cv, optimizer, optimizer_params,
                                             metrics, best_config_metric,
                                             eval_final_performance, use_test_set, test_size,
                                             project_folder, calculate_metrics_per_fold,
                                             calculate_metrics_across_folds,
                                             ignore_sanity_checks, random_seed, verbosity, learning_curves,
                                             learning_curves_cut, output_settings, performance_constraints,
                                             permutation_id, cache_folder, nr_of_processes, multi_threading,
                                             allow_multidim_targets)
