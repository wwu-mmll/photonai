from sklearn.datasets import load_diabetes
from photonai import RegressionPipe

my_pipe = RegressionPipe('diabetes',
                         best_config_metric='median_absolute_error',
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
