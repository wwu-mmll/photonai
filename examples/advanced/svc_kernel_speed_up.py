import os

from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Memory

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange


cache_dir = './tmp/kernel_cache'
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cachedir=cache_dir, verbose=0)


@memory.cache
def cached_rbf(X, Y):
    return rbf_kernel(X, Y)


# create toy data
n_features = 10000
n_samples = 1000
n_informative = 10
X, y = make_classification(n_samples, n_features, n_informative=n_informative)
gamma = 1 / n_features

"""
Especially with large datasets, it is unnecessary to recompute the kernel for every hyperparameter configuration.
For that reason, you can pass a cached kernel function that will only recompute the kernel if the input data changes.
If you don't want to cache the kernel, it still decreases the computation time by magnitudes when passing the kernel
as dedicated function. See this issue for details:  
https://github.com/scikit-learn/scikit-learn/issues/21410
https://stackoverflow.com/questions/69680420/using-a-custom-rbf-kernel-function-for-sklearns-svc-is-way-faster-than-built-in
"""
#kernel = 'kernel'
#kernel = rbf_kernel
kernel = cached_rbf

pipe = Hyperpipe('svm_with_custom_kernel',
                 inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 optimizer='sk_opt',
                 optimizer_params={'n_configurations': 15},
                 metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                 best_config_metric='accuracy',
                 project_folder='./tmp',
                 verbosity=1)

pipe += PipelineElement('StandardScaler')

pipe += PipelineElement('SVC',
                        hyperparameters={'C': FloatRange(1e-6, 1e6)},
                        gamma=gamma, kernel=kernel)

pipe.fit(X, y)

