
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.optimization.SpeedHacks import MinimumPerformance
from photonai.investigator.Investigator import Investigator

from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)

mongo_settings = PersistOptions(mongodb_connect_url="mongodb://localhost:27017/photon_db",
                                save_predictions=False,
                                save_feature_importances=False)


my_pipe = Hyperpipe('basic_svm_pipe_no_performance',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    calculate_metrics_across_folds=True,
                    eval_final_performance=False,
                    performance_constraints=[MinimumPerformance('accuracy', 0.96),
                                             MinimumPerformance('precision', 0.96)],  # skips next folds of inner cv if accuracy in first fold is below 0.96 etc.
                    verbosity=1,
                    persist_options=mongo_settings)

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('PCA', {'n_components': [5, 10, None]})
my_pipe += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

my_pipe.fit(X, y)


Investigator.show(my_pipe)

# Investigator.load_from_db(mongo_settings.mongodb_connect_url, my_pipe.name)

debug = True


