from sklearn.datasets import load_diabetes
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split

from photonai.base import Hyperpipe, PipelineElement

"""
Since PHOTONAI is built on top of the scikit-learn interface, 
it is possible to use direct functions from their package. 
Here the example of the feature importance via permutations. The example can be found at:
https://scikit-learn.org/stable/modules/permutation_importance.html
"""

diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(diabetes.data, diabetes.target, random_state=0)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')

my_pipe += PipelineElement("StandardScaler")
my_pipe += PipelineElement('Ridge', alpha=1e-2)
my_pipe.fit(X_train, y_train)

r = permutation_importance(my_pipe, X_val, y_val,
                           n_repeats=50,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
