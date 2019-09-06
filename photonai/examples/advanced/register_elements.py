from photonai.base import Hyperpipe, PipelineElement, OutputSettings, PhotonRegistry
from photonai.optimization import IntegerRange
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


# REGISTER ELEMENT
custom_elements_folder = './custom_elements'
registry = PhotonRegistry()
registry.register(photon_name='MyCustomEstimator',
                  class_str='custom_estimator.CustomEstimator',
                  element_type='Estimator',
                  custom_folder=custom_elements_folder)

registry.register(photon_name='MyCustomTransformer',
                  class_str='custom_transformer.CustomTransformer',
                  element_type='Transformer',
                  custom_folder=custom_elements_folder)

registry.activate(custom_elements_folder=custom_elements_folder)

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

settings = OutputSettings(project_folder='.')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('custom_estimator_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'k': 2},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
registry.list_available_elements()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
registry.info('MyCustomEstimator')
registry.info('MyCustomTransformer')


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('MyCustomEstimator')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True


