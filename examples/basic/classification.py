from photonai import ClassificationPipe
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit


X, y = load_breast_cancer(return_X_y=True)
my_pipe = ClassificationPipe(name='breast_cancer_analysis',
                             inner_cv=ShuffleSplit(n_splits=2),
                             scaling=True,
                             imputation=False,
                             imputation_nan_value=None,
                             feature_selection=False,
                             dim_reduction=True,
                             n_pca_components=10)
my_pipe.fit(X, y)
