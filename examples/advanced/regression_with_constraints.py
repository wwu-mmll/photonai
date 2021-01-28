from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import MinimumPerformanceConstraint, IntegerRange

X, y = load_boston(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(name='basic_svm_pipe_no_performance',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=3),
                    eval_final_performance=True,
                    verbosity=1,
<<<<<<< HEAD
                    project_folder='./result_folder',
                    output_settings=OutputSettings(mongodb_connect_url="mongodb://localhost:27017/photon_results",
                                                   save_output=True),
                    performance_constraints=[MinimumPerformance('mean_squared_error', 35, 'first'),
                                             MinimumPerformance('pearson_correlation', 0.7, 'all')])
=======
                    output_settings=OutputSettings(project_folder='./result_folder',
                                                   mongodb_connect_url="mongodb://localhost:27017/photon_results",
                                                   save_output=True,
                                                   plots=True),
                    performance_constraints=[MinimumPerformanceConstraint('mean_squared_error', 35, 'first'),
                                             MinimumPerformanceConstraint('pearson_correlation', 0.7, 'any')])
>>>>>>> develop

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# AND SHOW THE RESULTS IN THE WEBBASED PHOTON INVESTIGATOR TOOL
# Investigator.show(my_pipe)

# YOU CAN ALSO SAVE THE BEST PERFORMING PIPELINE FOR FURTHER USE
# my_pipe.save_optimum_pipe('/home/photon_user/photon_test/optimum_pipe.photon')
