from photonai import RegressionPipe, Hyperpipe, ClassificationPipe, OutputSettings, PipelineElement
from sklearn.datasets import load_diabetes


def create_ensemble(copy_pipe_fnc, outer_fold, best_config_outer_fold):
    optimum_pipe = copy_pipe_fnc()
    # set self to best config
    optimum_pipe.set_params(**best_config_outer_fold.config_dict)
    print("Delegate was here!")
    return optimum_pipe, best_config_outer_fold


my_pipe = RegressionPipe('diabetes',
                         project_folder='./tmp',
                         add_estimator=False,
                         ensemble_builder=create_ensemble,
                         cache_folder=None,  # important! caching does not work!
                         output_settings=OutputSettings(generate_best_model=False)  # important! do not change
                         )

my_pipe += PipelineElement('SVC')
# load data and train
X, y = load_diabetes(return_X_y=True)
my_pipe.fit(X, y)
