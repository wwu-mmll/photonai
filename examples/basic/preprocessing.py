from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import fetch_openml

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai.optimization import FloatRange

audiology = fetch_openml(name='audiology')
X = audiology.data.values
y = audiology.target.values

my_pipe = Hyperpipe('hot_encoder_pipeline',
                    inner_cv=ShuffleSplit(n_splits=5, test_size=0.2),
                    outer_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 20},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    allow_multidim_targets=True,
                    project_folder='./tmp')

pre_proc = Preprocessing()
pre_proc += PipelineElement('OneHotEncoder', sparse=False)
pre_proc += PipelineElement('LabelEncoder')
my_pipe += pre_proc

my_pipe += PipelineElement('PCA', hyperparameters={'n_components': FloatRange(0.2, 0.7)})
my_pipe += PipelineElement('SVC', hyperparameters={'C': FloatRange(1, 150)}, kernel='rbf')

my_pipe.fit(X, y)
