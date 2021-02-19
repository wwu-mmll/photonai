from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('multi_perceptron_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PhotonMLPClassifier', hyperparameters={'layer_1': IntegerRange(1, 5),
                                                                   'layer_2': IntegerRange(0, 5),
                                                                   'layer_3': IntegerRange(0, 5)})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
