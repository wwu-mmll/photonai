from photonai import ClassificationPipe, ClassifierSwitch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit


X, y = load_breast_cancer(return_X_y=True)
my_pipe = ClassificationPipe(name='breast_cancer_analysis',
                             inner_cv=ShuffleSplit(n_splits=2),
                             dim_reduction=True,
                             n_pca_components=10,
                             use_test_set=True)
my_pipe.fit(X, y)
