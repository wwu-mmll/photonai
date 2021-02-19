import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, PhotonRegistry
from photonai.optimization import IntegerRange

# REGISTER ELEMENT
base_folder = os.path.dirname(os.path.abspath(__file__))
custom_elements_folder = os.path.join(base_folder, '../advanced/custom_elements')

registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)
registry.register(photon_name='MyCustomEstimator',
                  class_str='custom_estimator.CustomEstimator',
                  element_type='Estimator')

registry.register(photon_name='MyCustomTransformer',
                  class_str='custom_transformer.CustomTransformer',
                  element_type='Transformer')

registry.activate()

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('custom_estimator_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 2},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
registry.list_available_elements()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
registry.info('MyCustomEstimator')
registry.info('MyCustomTransformer')

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)

my_pipe += PipelineElement('MyCustomEstimator')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
