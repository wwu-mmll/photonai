

from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from HPOFramework.HPOptimizers import GridSearchOptimizer
from TFLearnPipelineWrapper.TFDNNClassifier import TFDNNClassifier


class HyperpipeManager(BaseEstimator):

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer}

    def __init__(self, param_list, X, y=None, groups=None):

        self.param_list = param_list
        self.X = X
        self.y = y
        self.groups = groups
        self.cv = KFold(n_splits=10)
        self.cv_iter = list(self.cv.split(X, y, groups))

    def optimize(self, optimization_strategy):

        # 1. instantiate optimizer
        optimizer_class = self.OPTIMIZER_DICTIONARY[optimization_strategy]
        optimizer_instance = optimizer_class(self.param_list)

        # 2. iterate next_config
        for specific_config in optimizer_instance.next_config:
            hp = Hyperpipe(specific_config)
            config_loss = hp.calculate_cv_loss(self.X, self.y, self.cv_iter)
            print(config_loss)


class Hyperpipe(object):

    def __init__(self, specific_config, verbose=2, fit_params={}, error_score='raise'):

        self.params = specific_config

        # default
        self.return_train_score = True
        self.verbose = verbose
        self.fit_params = fit_params
        self.error_score = error_score

        #Todo: magically generate Pipeline from params, use: make_pipeline()
        # self.pipe = None

        pca = PCA()
        dnn = TFDNNClassifier(0.5)
        self.pipe = Pipeline(steps=[('pca', pca), ('dnn', dnn)])

    def calculate_cv_loss(self, X, y, cv_iter):
        scores = []
        for train, test in cv_iter:
            fit_and_predict_score = _fit_and_score(clone(self.pipe), X, y, self.score,
                                                   train, test, self.verbose, self.params,
                                                   fit_params=self.fit_params,
                                                   return_train_score=self.return_train_score,
                                                   return_n_test_samples=True,
                                                   return_times=True, return_parameters=True,
                                                   error_score=self.error_score)
            scores.append(fit_and_predict_score)
        return scores

    def score(self, estimator, X_test, y_test):
        return estimator.score(X_test, y_test)

