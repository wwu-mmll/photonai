from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE

my_pipe = Hyperpipe('sample_pairing_example_classification',
                    optimizer='grid_search',
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp',
                    random_seed=42123)


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SamplePairingClassification',
                           hyperparameters={'draw_limit': [500, 1000, 10000],
                                            'generator': Categorical(['nearest_pair'])},
                           distance_metric='euclidean',
                           test_disabled=True)

my_pipe += PipelineElement('RandomForestClassifier',
                           hyperparameters={'n_estimators': [10, 100]})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
