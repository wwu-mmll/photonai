from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class PhotonXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=100,
                 max_depth=3,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 gamma=0,
                 min_child_weight=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 objective='binary:logistic',
                 eval_metric='logloss',
                 n_jobs=None,
                 random_state=None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = None

    def fit(self, X, y):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            eval_metric=self.eval_metric,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
