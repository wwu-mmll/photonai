from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Stack, Branch, Switch, DataFilter
from photonai.optimization import FloatRange, IntegerRange

# LOAD DATA FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

my_pipe = Hyperpipe('data_integration',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 20},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='f1_score',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=0,
                    project_folder='./tmp/')

my_pipe += PipelineElement('SimpleImputer')
my_pipe += PipelineElement('StandardScaler', {}, with_mean=True)

# Use only "mean" features: [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness,
# mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension
mean_branch = Branch('MeanFeature')
mean_branch += DataFilter(indices=range(0, 10))
mean_branch += PipelineElement('SVC', {'C': FloatRange(0.1, 200)}, kernel='linear')

# Use only "error" features
error_branch = Branch('ErrorFeature')
error_branch += DataFilter(indices=range(10, 20))
error_branch += PipelineElement('SVC', {'C': FloatRange(0.1, 200)}, kernel='linear')

# use only "worst" features: [worst_radius, worst_texture, ..., worst_fractal_dimension]
worst_branch = Branch('WorstFeature')
worst_branch += DataFilter(indices=range(20, 30))
worst_branch += PipelineElement('SVC', {'C': FloatRange(0.1, 200)}, kernel='linear')

my_pipe += Stack('SourceStack', [mean_branch, error_branch, worst_branch])

my_pipe += Switch('EstimatorSwitch', [PipelineElement('RandomForestClassifier', {'n_estimators': IntegerRange(2, 5)}),
                                      PipelineElement('SVC')])

my_pipe.fit(X, y)
