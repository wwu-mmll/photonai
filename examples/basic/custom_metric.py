import numpy as np
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from keras.metrics import Accuracy

from photonai.base import Hyperpipe, PipelineElement


# you can have a simple delegate
def custom_metric(y_true, y_pred):
    def hot_encoding(targets, nclasses):
        """Convert indices to one-hot encoded labels."""
        targets = np.array(targets).reshape(-1)
        return np.eye(nclasses)[targets]
    return f1_score(hot_encoding(y_true, 3), hot_encoding(y_pred, 3), average='macro')


my_pipe = Hyperpipe('custom_metric_project',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    # and here is how to register it in photonai
                    metrics=[('custom_metric', custom_metric), Accuracy, 'accuracy'],
                    best_config_metric='custom_metric',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    allow_multidim_targets=True,
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SVC', kernel='rbf')

X, y = fetch_openml("cars1", return_X_y=True)
my_pipe.fit(X.values, y.values.astype(int))
