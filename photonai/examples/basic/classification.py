from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing, OutputSettings
from photonai.investigator import Investigator
from photonai.optimization import FloatRange, Categorical, IntegerRange


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    eval_final_performance=False,
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=OutputSettings(project_folder='./tmp/'))


preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))

# then do feature selection using a PCA,
my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': IntegerRange(5, 20)},
                           test_disabled=True)

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(0.5, 2)},
                           gamma='scale')

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

Investigator.show(my_pipe)


