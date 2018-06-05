
from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from photonai.optimization.Hyperparameters import FloatRange, Categorical

from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)

my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    write_to_db=True,
                    mongodb_connect_url="mongodb://localhost:27017/photon_db")


my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('PCA', {'n_components': [5, 10, None]})
my_pipe += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

my_pipe.fit(X, y)

debug = True
