from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import Categorical

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
settings = OutputSettings(project_folder='.')
my_pipe = Hyperpipe('sample_pairing_example_classification',
                    optimizer='grid_search',
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# add sample pairing
my_pipe += PipelineElement('SamplePairingClassification', {'draw_limit': [500, 1000, 10000],
                                                       'generator': Categorical(['nearest_pair', 'random_pair'])},
                           distance_metric='euclidean', test_disabled=True)

# engage and optimize a Random Forest Classifier
my_pipe += PipelineElement('RandomForestClassifier', hyperparameters={'n_estimators': [10]})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True


