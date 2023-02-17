import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import IntegerRange, MinimumPerformanceConstraint

# setup training and test workflow
my_pipe = Hyperpipe('heart_failure_final_perf',
                    outer_cv=ShuffleSplit(n_splits=100, test_size=0.2),
                    inner_cv=KFold(n_splits=10, shuffle=True),
                    use_test_set=True,
                    metrics=['balanced_accuracy', 'f1_score', 'matthews_corrcoef',
                             'sensitivity', 'specificity'],
                    best_config_metric='f1_score',
                    optimizer='grid_search',
                    project_folder='./tmpv2',
                    performance_constraints=MinimumPerformanceConstraint('f1_score', threshold=0.7, strategy="mean"),
                    cache_folder='./cache',
                    verbosity=0)


# arrange a sequence of algorithms subsequently applied
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SimpleImputer')


my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name': ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE']},
                           test_disabled=True)

# compare different learning algorithms in an OR_Element
my_pipe += PipelineElement('RandomForestClassifier', criterion='gini', bootstrap=True,
                           hyperparameters={'min_samples_split': IntegerRange(2, 30),
                                            'max_features': ['auto', 'sqrt', 'log2']})

# read data
df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# start the training, optimization and test procedure
my_pipe.fit(X, y)

debug = True
