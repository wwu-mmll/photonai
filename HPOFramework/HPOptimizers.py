
from sklearn.model_selection._search import ParameterGrid
# , BaseSearchCV


# class GridSearchOptimizer(BaseSearchCV):
#
#     def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
#                  n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
#                  pre_dispatch='2*n_jobs', error_score='raise',
#                  return_train_score=True):
#
#         super(GridSearchOptimizer, self).__init__(
#             estimator=estimator, scoring=scoring, fit_params=fit_params,
#             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
#             pre_dispatch=pre_dispatch, error_score=error_score,
#             return_train_score=return_train_score)
#
#         self.param_grid = param_grid
#         self.parameter_iterable = ParameterGrid(self.param_grid)
#
#         #Todo: _check_param_grid(param_grid)

class GridSearchOptimizer(object):
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.parameter_iterable = ParameterGrid(self.param_grid)
        self.next_config = self.next_config_generator()

    def next_config_generator(self):
        for parameters in self.parameter_iterable:
            yield parameters

    def evaluate_recent_performance(self, config, performance):
        # influence return value of next_config
        pass

