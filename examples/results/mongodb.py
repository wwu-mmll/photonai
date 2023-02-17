from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import FloatRange, Categorical, IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(mongodb_connect_url="mongodb://localhost:27017/photon_results")

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_MongoDB_pipe',  # the name of your pipeline
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    project_folder='./tmp/',
                    # append the output_settings argument to hand over the mongo_setting to your pipeline
                    output_settings=mongo_settings)

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2, num=10)}, gamma='scale')

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
