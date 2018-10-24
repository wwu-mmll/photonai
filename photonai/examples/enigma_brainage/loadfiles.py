import pandas
import numpy as np
from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


TRAIN = False

# data
file_name = 'AllFS'
file_name_2 = 'Age'
setID_train = 'train'
setID_test = 'test'


if TRAIN:

    # load trainings_data
    X_tr = pandas.read_csv(file_name + '_' + setID_train + 'ControlsFemales_tt.txt', header=None, delimiter='\t').values
    y_tr = np.squeeze(pandas.read_csv(file_name_2 + '_' + setID_train + 'ControlsFemales_tt.txt', header=None, delimiter='\t').values)

    # create hyperpipe
    my_pipe = Hyperpipe('enigma_brainage',  # the name of your pipeline
                        optimizer='grid_search',  # which optimizer PHOTON shall use
                        metrics=['mean_squared_error', 'mean_absolute_error'],  # the performance metrics of your interest
                        best_config_metric='mean_absolute_error',  # after hyperparameter search, the metric declares the winner config
                        inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively
                        verbosity=1)  # get error, warn and info message                    )

    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('RandomForestRegressor', {'min_samples_leaf': [1, 2, 3], 'min_samples_split': [2, 4, 6]})

    # fit & save
    my_pipe.fit(X_tr, y_tr)
    my_pipe.save_optimum_pipe('enigma_brainage.photon')

else:
    my_pipe = Hyperpipe.load_optimum_pipe('enigma_brainage.photon')

# load test data
X_te = pandas.read_csv(file_name + '_' + setID_test + 'ControlsFemales_tt.txt', header=None, delimiter='\t').values
y_te = np.squeeze(pandas.read_csv(file_name_2 + '_' + setID_test + 'ControlsFemales_tt.txt', header=None, delimiter='\t').values)

# predict test data
y_pred = my_pipe.predict(X_te)

test_error = mean_absolute_error(y_te, y_pred)
print("Mean absolute error testset")
print(test_error)

debug = True