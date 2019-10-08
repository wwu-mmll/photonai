from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('PhotonMLPClassifier', hyperparameters={'layer_1': IntegerRange(0, 5),
                                                                   'layer_2': IntegerRange(0, 5),
                                                                   'layer_3': IntegerRange(0, 5)})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True


