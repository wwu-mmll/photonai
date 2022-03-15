from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

from photonai.base import Hyperpipe, PipelineElement

diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(diabetes.data, diabetes.target, random_state=0)

my_pipe = Hyperpipe('basic_ridge_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error',
                    project_folder='./tmp')

my_pipe += PipelineElement("StandardScaler")
my_pipe += PipelineElement('Ridge', alpha=1e-2)
my_pipe.fit(X_train, y_train)

r = my_pipe.get_permutation_feature_importances(n_repeats=50, random_state=0)

for i in r["mean"].argsort()[::-1]:
    if r["mean"][i] - 2 * r["std"][i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
              f"{r['mean'][i]:.3f}"
              f" +/- {r['std'][i]:.3f}")


# get permutation importances posthoc
# reloaded_hyperpipe = Hyperpipe.reload_hyperpipe("full_path/to/results_folder/", X_train, y_train)
# post_hoc_perm_importances = Hyperpipe.get_permutation_feature_importances(n_repeats=5, random_state=0)