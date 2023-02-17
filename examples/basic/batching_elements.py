from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai.optimization import FloatRange, Categorical, IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('batching_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    project_folder='./tmp/',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=0)


preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder", batch_size=10)

my_pipe += preprocessing

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 10)}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 1)}, gamma='scale')

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True
