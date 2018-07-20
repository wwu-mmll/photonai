from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.validation.PermutationTest import PermutationTest

from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


def create_hyperpipe():
    mongo_settings = PersistOptions(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_tests",
                                    save_predictions='best',
                                    save_feature_importances='best')

    my_pipe = Hyperpipe('basic_svm_pipe',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=KFold(n_splits=3),
                        inner_cv=KFold(n_splits=3),
                        calculate_metrics_across_folds=True,
                        eval_final_performance=True,
                        verbosity=1, persist_options=mongo_settings)

    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                       'C': FloatRange(0.5, 2, "linspace", num=3)})
    return my_pipe

X, y = load_breast_cancer(True)
perm_tester = PermutationTest(create_hyperpipe, n_perms=20, n_processes=3, random_state=11)
perm_tester.fit(X, y)
