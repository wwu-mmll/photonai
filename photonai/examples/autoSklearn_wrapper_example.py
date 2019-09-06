from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import FloatRange, Categorical
from photonai.investigator import Investigator
from photonai.base.register import PhotonRegister
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_boston(True)

# setup Hyperpipe
pipe_name = 'autoSklearn_pipe'
pers_opts = OutputSettings(local_file=pipe_name + '.p',
                           save_predictions='best', save_feature_importances='None')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(pipe_name, optimizer='grid_search',
                    metrics=['mean_absolute_error', 'pearson_correlation'],
                    best_config_metric='mean_absolute_error',
                    #outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=2),
                    output_settings=pers_opts,
                    verbosity=1)


# Register AutoSklearnRegressor
# PhotonRegister.save(photon_name='AutoSklearnRegressor', photon_package='PhotonCore',
#                     class_str='photonai.modelwrapper.AutoSklearnRegressor.AutoSklearnRegressor', element_type='Estimator')
#PhotonRegister.info('AutoSklearnRegressor')

# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('AutoSklearnRegressor', time_left_for_this_task=80, per_run_time_limit=10)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# AND SHOW THE RESULTS IN THE WEBBASED PHOTON INVESTIGATOR TOOL
#Investigator.show(my_pipe)

# YOU CAN ALSO SAVE THE BEST PERFORMING PIPELINE FOR FURTHER USE
#my_pipe.save_optimum_pipe('optimum_pipe.photon')

# YOU CAN ALSO LOAD YOUR RESULTS FROM THE MONGO DB
# Investigator.load_from_db(mongo_settings.mongodb_connect_url, my_pipe.name)

debug = True
