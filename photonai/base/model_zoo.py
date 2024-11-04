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
    """The PHOTONAI ClassifierSwitch creates an instance of a PHOTONAI Switch with classification-specific estimators
    and pre-defined hyperparameter ranges.

        The ClassifierSwitch contains:
            - an SVC
                - C: [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8]
                - kernel: ['rbf', 'linear'],
                - max_iter=1000
            - a RandomForestClassifier
                - max_features: ["sqrt", "log2"]
                - min_samples_leaf: [0.01, 0.1, 0.2]
            - an AdaBoostClassifier
                - n_estimators: [10, 25, 50]
            - LogisticRegression
                - C: [1e-4, 1e-2, 1, 1e2, 1e4]
                - penalty: ['l1', 'l2']
            - GaussianNB
            - KNeighborsClassifier
                - n_neighbors: [5, 10, 15]

              Attributes:

                  elements (list):
                      Contains `all PipelineElement or Hyperpipe objects that are added to the pipeline.
                      The RegressorSwitch contains an SVR, a RandomForestRegressor, LinearRegression, an AdaBoostRegressor
                      and KNeighborsRegressor.

              Example:
                  ``` python
                     from photonai import ClassificationPipe, ClassifierSwitch
                     from sklearn.datasets import load_breast_cancer

                     X, y = load_breast_cancer(return_X_y=True)
                     my_pipe = ClassificationPipe(name='breast_cancer_analysis',
                                                  add_estimator=False)
                     my_pipe += ClassifierSwitch()
                     # The default estimator is a ClassifierSwitch containing an SVC, a RandomForestClassifier,
                     # an AdaBoostClassifier, LogisticRegression, GaussianNB and KNeighborsClassifier.
                     my_pipe.fit(X, y)
                  ```
                  """

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
    """The PHOTONAI RegressorSwitch creates an instance of a PHOTONAI Switch with regression-specific estimators and
    pre-defined hyperparameter ranges.

    The RegressorSwitch contains:
        - an SVR
            - C: [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8]
            - kernel: ['rbf', 'linear'],
            - max_iter=1000
        - a RandomForestRegressor
            - max_features: ["sqrt", "log2"]
            - min_samples_leaf: [0.01, 0.1, 0.2]
        - an AdaBoostRegressor
            - n_estimators: [10, 25, 50]
        - LinearRegression
        - KNeighborsRegressor
            - n_neighbors: [5, 10, 15]


          Attributes:

              elements (list):
                  Contains `all PipelineElement or Hyperpipe objects that are added to the pipeline.
                  The RegressorSwitch contains an SVR, a RandomForestRegressor, LinearRegression, an AdaBoostRegressor
                  and KNeighborsRegressor.

          Example:
              ``` python
                from sklearn.datasets import load_diabetes
                from photonai import RegressionPipe, RegressorSwitch

                my_pipe = RegressionPipe('diabetes',
                                         add_estimator=False)
                # load data and train
                X, y = load_diabetes(return_X_y=True)
                my_pipe += RegressorSwitch()
                # The RegressorSwitch contains an SVR, a RandomForestRegressor, LinearRegression, an AdaBoostRegressor
                # and KNeighborsRegressor.
                my_pipe.fit(X, y)

              ```
              """

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
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=10,
                                                                                                shuffle=True,
                                                                                                random_state=42),
                 outer_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = KFold(n_splits=5,
                                                                                                shuffle=True,
                                                                                                random_state=42),
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

    def fit(self, X=None, y=None, **kwargs):
        if (X is not None and self.X_csv_path is not None) or (y is not None and self.y_csv_path is not None):
            raise ValueError("You can either give the fit function data or the pipe definition paths "
                             "to csv files to load data from. Not both.")

        if (X is None and self.X_csv_path is None) or (y is None and self.y_csv_path is None):
            raise ValueError("Please give either the fit function array-like X and y data "
                             "or give the pipe definition paths to csv files to load X and y data from.")

        X = X if X is not None else pd.read_csv(self.X_csv_path, delimiter=self.delimiter)
        y = y if y is not None else pd.read_csv(self.y_csv_path, delimiter=self.delimiter)
        super().fit(X, y, **kwargs)


class ClassificationPipe(DefaultPipeline):
    """The PHOTONAI ClassificationPipe creates an instance of a machine learning pipeline with classification-
    specific defaults, such as classification metrics. If not specified otherwise, a default pipeline is created
     consisting of a standard scaler, dimensionality reduction via PCA and an estimator switch, i.e. an OR-element
     comparing different learning algorithms for that pipeline.

     The user can prevent the creation of a default pipeline, and instead add a custom pipeline: So called PHOTONAI
     PipelineElements can be added, each of them being a data-processing method or a learning algorithm.
     By choosing and combining data-processing methods or algorithms, and arranging them with the PHOTONAI classes,
    simple and complex pipeline architectures can be designed rapidly.

    The PHOTONAI ClassificationPipe is a child of the Hyperpipe class and as such, automatizes the nested training,
    test and hyperparameter optimization procedures.

    The Hyperpipe monitors:
      - the nested-cross-validated training and test procedure,
      - communicates with the hyperparameter optimization strategy,
      - streams information between the pipeline elements,
      - logs all results obtained and evaluates the performance,
      - guides the hyperparameter optimization process by a so-called best config metric which is used to select
        the best performing hyperparameter configuration.

      Attributes:
          optimum_pipe (PhotonPipeline):
              An sklearn pipeline object that is fitted to the training data
              according to the best hyperparameter configuration found.
              Currently, we don't create an ensemble of all best hyperparameter
              configs over all folds. We find the best config by comparing
              the test error across outer folds. The hyperparameter config of the best
              fold is used as the optimal model and is then trained on the complete set.

          best_config (dict):
              Dictionary containing the hyperparameters of the
              best configuration. Contains the parameters in the sklearn
              interface of model_name__parameter_name: parameter value.

          results (MDBHyperpipe):
              Object containing all information about the for the
              performed hyperparameter search. Holds the training and test
              metrics for all outer folds, inner folds
              and configurations, as well as additional information.

          elements (list):
              Contains `all PipelineElement or Hyperpipe
              objects that are added to the pipeline.

      Example:
          ``` python
            from photonai import ClassificationPipe
            from sklearn.datasets import load_breast_cancer

            X, y = load_breast_cancer(return_X_y=True)
            my_pipe = ClassificationPipe(name='breast_cancer_analysis',
                                         add_default_pipeline_elements=True,
                                         scaling=True,
                                         imputation=False,
                                         imputation_nan_value=None,
                                         feature_selection=False,
                                         dim_reduction=True,
                                         n_pca_components=10,
                                         add_estimator=True)
            my_pipe.fit(X, y)
            # The default estimator is a ClassifierSwitch containing an SVC, a RandomForestClassifier,
            #                an AdaBoostClassifier, LogisticRegression, GaussianNB and KNeighborsClassifier.

          ```

      """

    def __init__(self,
                 name: Optional[str] = 'classification_pipeline',
                 project_folder: Union[str, Path] = './classification/',
                 metrics: list = None,
                 best_config_metric: str = 'balanced_accuracy',
                 default_estimator: Union[BaseEstimator, ClassifierSwitch] = ClassifierSwitch,
                 **kwargs):
        """
                Parameters:
                       name:
                           Name of RegressionPipe instance.

                       X_csv_path:
                           A path or string to a csv_file, which contains the feature values. If given, then the features
                           are loaded from that file in the fit function.
                           Default is None.

                       y_csv_path:
                           A path or string to a csv_file, which contains the target values. If given, then the target values
                           are loaded from that file in the fit function.
                           Default is None.

                       delimiter:
                           The delimiter that is used in both the csv files from X_csv_path and y_csv_path.
                           Default is ",".

                       add_default_pipeline_elements (bool):
                           If a default pipeline shall be generated. The default pipeline consists of
                            scaling with a StandardScaler, dimensionality reduction via PCA and an RegressionSwitch.
                            The default pipeline can be customized by the parameters
                            - scaling = True/False
                            - imputation = True/False
                            - imputation_nan_value = np.nan
                            - feature_selection = True/False
                            - dim_reduction = True/False
                            - n_pca_components = int, float, IntegerRange, FloatRange, None
                            - add_estimator = True/False
                            - default_estimator: BaseEstimator = RegressionSwitch

                        scaling:
                           If True, a StandardScaler is added to the default pipeline. Only relevant if parameter
                           add_default_pipelne is True.
                           Default is True

                        imputation:
                           If True, a SimpleImputer is added to the default pipeline. Only relevant if parameter
                               add_default_pipelne is True.
                           Default is False.

                        imputation_nan_value:
                           If imputation is set to True, this parameter defines the value that defines missing items.
                            Default is np.nan.

                        feature_selection:
                           If True, a feature selection element is added to the default pipeline. Only relevant if parameter
                               add_default_pipelne is True.
                           Default is False.

                        dim_reduction:
                           If True, a PCA is added to the default pipeline. Only relevant if parameter add_default_pipeline is
                           True.
                           Default is True.

                        n_pca_components:
                           If dim_reduction is True, this parameter defines the number of PCA components to be kept.
                           It can be an Integer, Float, an IntegerRange, FloatRange or None.
                           Default is None.

                        add_estimator:
                           If True, an estimator which is defined in default_estimator is added to the pipeline. Only relevant
                           if parameter add_default_pipelne is True.
                           Default is True.

                        default_estimator:
                           If parameter add_estimator is True, then an object of the type defined by default_estimator
                           is added to the pipeline. Use a string keyword argument such as 'SVC' or 'SVR'
                           or photon native constructs such as PipelineElement, Switch or Stack.

                           Default is ClassifierSwitch containing an SVC, a RandomForestClassifier,
                           an AdaBoostClassifier, LogisticRegression, GaussianNB and KNeighborsClassifier.

                       inner_cv:
                           Cross validation strategy to test hyperparameter configurations, generates the validation set.
                           Default is KFold(n_splits=10, shuffle=True, random_state=42).

                       outer_cv:
                           Cross validation strategy to use for the hyperparameter search itself, generates the test set.
                           Default is KFold(n_splits=5, shuffle=True, random_state=42).

                       optimizer:
                           Hyperparameter optimization algorithm.

                           - In case a string literal is given:
                               - "grid_search": Optimizer that iteratively tests all possible hyperparameter combinations.
                               - "random_grid_search": A variation of the grid search optimization that randomly picks
                                   hyperparameter combinations from all possible hyperparameter combinations.
                               - "sk_opt": Scikit-Optimize based on theories of bayesian optimization.
                               - "random_search": randomly chooses hyperparameter from grid-free domain.
                               - "smac": SMAC based on theories of bayesian optimization.
                               - "nevergrad": Nevergrad based on theories of evolutionary learning.

                           - In case an object is given:
                               expects the object to have the following methods:
                               - `ask`: returns a hyperparameter configuration in form of an dictionary containing
                                   key->value pairs in the sklearn parameter encoding `model_name__parameter_name: parameter_value`
                               - `prepare`: takes a list of pipeline elements and their particular hyperparameters to prepare the
                                            hyperparameter space
                               - `tell`: gets a tested config and the respective performance in order to
                                   calculate a smart next configuration to process

                           Default is random_search.

                       optimizer_params:
                           Define parameters of the optimizer. Given as dict.
                           Default is {'n_configurations': 25}

                       metrics:
                           Metrics that should be calculated for both training, validation and test set
                           Use the pre-imported metrics from sklearn and photonai, or register your own

                           - Metrics for `classification`:
                               - `accuracy`: sklearn.metrics.accuracy_score
                               - `matthews_corrcoef`: sklearn.metrics.matthews_corrcoef
                               - `confusion_matrix`: sklearn.metrics.confusion_matrix,
                               - `f1_score`: sklearn.metrics.f1_score
                               - `hamming_loss`: sklearn.metrics.hamming_loss
                               - `log_loss`: sklearn.metrics.log_loss
                               - `precision`: sklearn.metrics.precision_score
                               - `recall`: sklearn.metrics.recall_score
                           - Other metrics
                               - `pearson_correlation`: photon_core.framework.Metrics.pearson_correlation
                               - `variance_explained`:  photon_core.framework.Metrics.variance_explained_score
                               - `categorical_accuracy`: photon_core.framework.Metrics.categorical_accuracy_score

                           Default are ['balanced_accuracy', 'specificity', 'precision', 'recall', 'f1_score',
                                        'matthews_corrcoef'].

                       best_config_metric:
                           The metric that should be maximized or minimized in order to choose
                           the best hyperparameter configuration.

                           Default is 'balanced_accuracy'.

                       eval_final_performance:
                           DEPRECATED! Use "use_test_set" instead!

                       use_test_set:
                           If the metrics should be calculated for the test set,
                           otherwise the test set is seperated but not used.

                           Default is True.

                       project_folder:
                           The output folder in which all files generated by the
                           PHOTONAI project are saved to.

                       test_size:
                           The amount of the data that should be left out if no outer_cv is given and
                           eval_final_performance is set to True.

                           Default is 0.2.

                       calculate_metrics_per_fold:
                           If True, the metrics are calculated for each inner_fold.
                           If False, calculate_metrics_across_folds must be True.

                       calculate_metrics_across_folds:
                           If True, the metrics are calculated across all inner_fold.
                           If False, calculate_metrics_per_fold must be True.

                       ignore_sanity_checks:
                           If True, photonai will not verify use cases such as:
                               - classification, imbalanced classes and best_config_metric set to "accuracy"

                       random_seed:
                           Random Seed. This is also given to numpy and optimizer libraries.

                       verbosity:
                           The level of verbosity, 0 is least talkative and
                           gives only warn and error, 1 gives adds info and 2 adds debug.

                       learning_curves:
                           Enables learning curve procedure. Evaluate learning process over
                           different sizes of input. Depends on learning_curves_cut.

                       learning_curves_cut:
                           The tested relative cuts for data size.

                       performance_constraints:
                           Objects that indicate whether a configuration should
                           be tested further. For example, the inner fold of a config
                           does not perform better than the dummy performance.

                       permutation_id:
                           String identifier for permutation tests.

                       cache_folder:
                           Folder path for multi-processing.

                       nr_of_processes:
                           Determined the amount of simultaneous calculation of outer_folds.

                       multi_threading:
                           If true dask is used in multi threading mode, if false multi processing

                       allow_multidim_targets:
                           Allows multidimensional targets.

               """

        metrics = metrics if metrics is not None else ['balanced_accuracy',
                                                       'specificity',
                                                       'precision',
                                                       'recall',
                                                       'f1_score',
                                                       'matthews_corrcoef']

        super(ClassificationPipe, self).__init__(name=name,
                                                 project_folder=project_folder,
                                                 metrics=metrics,
                                                 best_config_metric=best_config_metric,
                                                 default_estimator=default_estimator,
                                                 **kwargs)


class RegressionPipe(DefaultPipeline):
    """The PHOTONAI RegressionPipe creates an instance of a machine learning pipeline with regression-specific defaults,
    such as the most common regression metrics. If not specified otherwise, a default pipeline is created
         consisting of a standard scaler, dimensionality reduction via PCA and an estimator switch, i.e. an OR-element
         comparing different learning algorithms for that pipeline.

         The user can prevent the creation of a default pipeline, and instead add a custom pipeline: So called PHOTONAI
         PipelineElements can be added, each of them being a data-processing method or a learning algorithm.
         By choosing and combining data-processing methods or algorithms, and arranging them with the PHOTONAI classes,
        simple and complex pipeline architectures can be designed rapidly.

        The PHOTONAI RegressionPipe is a child of the Hyperpipe class and as such, automatizes the nested training,
        test and hyperparameter optimization procedures.

        The Hyperpipe monitors:
          - the nested-cross-validated training and test procedure,
          - communicates with the hyperparameter optimization strategy,
          - streams information between the pipeline elements,
          - logs all results obtained and evaluates the performance,
          - guides the hyperparameter optimization process by a so-called best config metric which is used to select
            the best performing hyperparameter configuration.

          Attributes:
              optimum_pipe (PhotonPipeline):
                  An sklearn pipeline object that is fitted to the training data
                  according to the best hyperparameter configuration found.
                  Currently, we don't create an ensemble of all best hyperparameter
                  configs over all folds. We find the best config by comparing
                  the test error across outer folds. The hyperparameter config of the best
                  fold is used as the optimal model and is then trained on the complete set.

              best_config (dict):
                  Dictionary containing the hyperparameters of the
                  best configuration. Contains the parameters in the sklearn
                  interface of model_name__parameter_name: parameter value.

              results (MDBHyperpipe):
                  Object containing all information about the for the
                  performed hyperparameter search. Holds the training and test
                  metrics for all outer folds, inner folds
                  and configurations, as well as additional information.

              elements (list):
                  Contains `all PipelineElement or Hyperpipe
                  objects that are added to the pipeline.

          Example:
              ``` python
                from sklearn.datasets import load_diabetes
                from photonai import RegressionPipe

                my_pipe = RegressionPipe('diabetes',
                                         add_default_pipeline_elements=True,
                                         scaling=True,
                                         imputation=False,
                                         imputation_nan_value=None,
                                         feature_selection=False,
                                         dim_reduction=True,
                                         n_pca_components=10,
                                         add_estimator=True)
                # load data and train
                X, y = load_diabetes(return_X_y=True)
                my_pipe.fit(X, y)
                # Default is RegressorSwitch, containing an SVR, a RandomForestRegressor, LinearRegression,
                # an AdaBoostRegressor and KNeighborsRegressor.

              ```
          """

    def __init__(self,
                 name: Optional[str] = 'classification_pipeline',
                 project_folder: Union[str, Path] = './regression/',
                 metrics: list = None,
                 best_config_metric: str = 'mean_squared_error',
                 default_estimator: Union[BaseEstimator, RegressorSwitch] = RegressorSwitch,
                 **kwargs):
        """
         Parameters:
                name:
                    Name of RegressionPipe instance.

                X_csv_path:
                    A path or string to a csv_file, which contains the feature values. If given, then the features
                    are loaded from that file in the fit function.
                    Default is None.

                y_csv_path:
                    A path or string to a csv_file, which contains the target values. If given, then the target values
                    are loaded from that file in the fit function.
                    Default is None.

                delimiter:
                    The delimiter that is used in both the csv files from X_csv_path and y_csv_path.
                    Default is ",".

                add_default_pipeline_elements (bool):
                    If a default pipeline shall be generated. The default pipeline consists of
                     scaling with a StandardScaler, dimensionality reduction via PCA and an RegressionSwitch.
                     The default pipeline can be customized by the parameters
                     - scaling = True/False
                     - imputation = True/False
                     - imputation_nan_value = np.nan
                     - feature_selection = True/False
                     - dim_reduction = True/False
                     - n_pca_components = int, float, IntegerRange, FloatRange, None
                     - add_estimator = True/False
                     - default_estimator: BaseEstimator = RegressionSwitch

                 scaling:
                    If True, a StandardScaler is added to the default pipeline. Only relevant if parameter
                    add_default_pipelne is True.
                    Default is True

                 imputation:
                    If True, a SimpleImputer is added to the default pipeline. Only relevant if parameter
                        add_default_pipelne is True.
                    Default is False.

                 imputation_nan_value:
                    If imputation is set to True, this parameter defines the value that defines missing items.
                     Default is np.nan.

                 feature_selection:
                    If True, a feature selection element is added to the default pipeline. Only relevant if parameter
                        add_default_pipelne is True.
                    Default is False.

                 dim_reduction:
                    If True, a PCA is added to the default pipeline. Only relevant if parameter add_default_pipeline is
                    True.
                    Default is True.

                 n_pca_components:
                    If dim_reduction is True, this parameter defines the number of PCA components to be kept.
                    It can be an Integer, Float, an IntegerRange, FloatRange or None.
                    Default is None.

                 add_estimator:
                    If True, an estimator which is defined in default_estimator is added to the pipeline. Only relevant
                    if parameter add_default_pipelne is True.
                    Default is True.

                 default_estimator:
                    If parameter add_estimator is True, then an object of the type defined by default_estimator
                    is added to the pipeline. Use a string keyword argument such as 'SVC' or 'SVR'
                    or photon native constructs such as PipelineElement, Switch or Stack.

                    Default is RegressorSwitch, containing an SVR, a RandomForestRegressor, LinearRegression,
                    an AdaBoostRegressor and KNeighborsRegressor.

                inner_cv:
                    Cross validation strategy to test hyperparameter configurations, generates the validation set.
                    Default is KFold(n_splits=10, shuffle=True, random_state=42).

                outer_cv:
                    Cross validation strategy to use for the hyperparameter search itself, generates the test set.
                    Default is KFold(n_splits=5, shuffle=True, random_state=42).

                optimizer:
                    Hyperparameter optimization algorithm.

                    - In case a string literal is given:
                        - "grid_search": Optimizer that iteratively tests all possible hyperparameter combinations.
                        - "random_grid_search": A variation of the grid search optimization that randomly picks
                            hyperparameter combinations from all possible hyperparameter combinations.
                        - "sk_opt": Scikit-Optimize based on theories of bayesian optimization.
                        - "random_search": randomly chooses hyperparameter from grid-free domain.
                        - "smac": SMAC based on theories of bayesian optimization.
                        - "nevergrad": Nevergrad based on theories of evolutionary learning.

                    - In case an object is given:
                        expects the object to have the following methods:
                        - `ask`: returns a hyperparameter configuration in form of an dictionary containing
                            key->value pairs in the sklearn parameter encoding `model_name__parameter_name: parameter_value`
                        - `prepare`: takes a list of pipeline elements and their particular hyperparameters to prepare the
                                     hyperparameter space
                        - `tell`: gets a tested config and the respective performance in order to
                            calculate a smart next configuration to process

                    Default is random_search.

                optimizer_params:
                    Define parameters of the optimizer. Given as dict.
                    Default is {'n_configurations': 25}

                metrics:
                    Metrics that should be calculated for both training, validation and test set
                    Use the pre-imported metrics from sklearn and photonai, or register your own

                    - Metrics for `regression`:
                        - `mean_squared_error`: sklearn.metrics.mean_squared_error
                        - `mean_absolute_error`: sklearn.metrics.mean_absolute_error
                        - `explained_variance`: sklearn.metrics.explained_variance_score
                        - `r2`: sklearn.metrics.r2_score
                    - Other metrics
                        - `pearson_correlation`: photon_core.framework.Metrics.pearson_correlation
                        - `variance_explained`:  photon_core.framework.Metrics.variance_explained_score
                        - `categorical_accuracy`: photon_core.framework.Metrics.categorical_accuracy_score

                    Default are ['mean_absolute_error', 'mean_squared_error', 'explained_variance']

                best_config_metric:
                    The metric that should be maximized or minimized in order to choose
                    the best hyperparameter configuration.

                    Default is 'mean_squared_error'.

                eval_final_performance:
                    DEPRECATED! Use "use_test_set" instead!

                use_test_set:
                    If the metrics should be calculated for the test set,
                    otherwise the test set is seperated but not used.

                    Default is True.

                project_folder:
                    The output folder in which all files generated by the
                    PHOTONAI project are saved to.

                test_size:
                    The amount of the data that should be left out if no outer_cv is given and
                    eval_final_performance is set to True.

                    Default is 0.2.

                calculate_metrics_per_fold:
                    If True, the metrics are calculated for each inner_fold.
                    If False, calculate_metrics_across_folds must be True.

                calculate_metrics_across_folds:
                    If True, the metrics are calculated across all inner_fold.
                    If False, calculate_metrics_per_fold must be True.

                ignore_sanity_checks:
                    If True, photonai will not verify use cases such as:
                        - classification, imbalanced classes and best_config_metric set to "accuracy"

                random_seed:
                    Random Seed. This is also given to numpy and optimizer libraries.

                verbosity:
                    The level of verbosity, 0 is least talkative and
                    gives only warn and error, 1 gives adds info and 2 adds debug.

                learning_curves:
                    Enables learning curve procedure. Evaluate learning process over
                    different sizes of input. Depends on learning_curves_cut.

                learning_curves_cut:
                    The tested relative cuts for data size.

                performance_constraints:
                    Objects that indicate whether a configuration should
                    be tested further. For example, the inner fold of a config
                    does not perform better than the dummy performance.

                permutation_id:
                    String identifier for permutation tests.

                cache_folder:
                    Folder path for multi-processing.

                nr_of_processes:
                    Determined the amount of simultaneous calculation of outer_folds.

                multi_threading:
                    If true dask is used in multi threading mode, if false multi processing

                allow_multidim_targets:
                    Allows multidimensional targets.

        """

        metrics = metrics if metrics is not None else ['mean_absolute_error',
                                                       'mean_squared_error',
                                                       'explained_variance']

        super(RegressionPipe, self).__init__(name=name,
                                             project_folder=project_folder,
                                             metrics=metrics,
                                             best_config_metric=best_config_metric,
                                             default_estimator=default_estimator,
                                             **kwargs)
