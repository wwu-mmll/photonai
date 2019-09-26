from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.investigator import Investigator
from photonai.optimization import FloatRange, Categorical

# loading the iris dataset
X, y = load_iris(True)

settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('multiclass_svm_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'k': 10},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=3, shuffle=True),
                    verbosity=1,
                    output_settings=settings)

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2)}, gamma='scale')
# my_pipe += PipelineElement('DecisionTreeClassifier')

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
Investigator.show(my_pipe)
debug = True
