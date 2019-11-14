import warnings

import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml, fetch_olivetti_faces
import os
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch
from photonai.optimization import Categorical, IntegerRange, FloatRange
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#X,y = fetch_openml(data_id=554, return_X_y=True)
dataset = fetch_olivetti_faces(download_if_missing=True)
X = dataset["data"]
y = dataset["target"]

# DEFINE OUTPUT SETTINGS
base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache")
cache_grid_folder_path = os.path.join(base_folder, "cache_grid")
tmp_folder_path = os.path.join(base_folder, "tmp")

settings = OutputSettings(project_folder=tmp_folder_path)

time_limit_secondes = 60*15

# DESIGN TWO PIPELINES
smac_pipe = Hyperpipe('Olivetti_smac',
                      optimizer='smac',
                      optimizer_params={'wallclock_limit': time_limit_secondes, 'run_limit':20},
                      metrics=['accuracy'],
                      best_config_metric='accuracy',
                      inner_cv=KFold(n_splits=3),
                      cache_folder=cache_folder_path,
                      verbosity=0)

grid_pipe = Hyperpipe('Olivetti_grid',
                      optimizer='timeboxed_random_grid_search',
                      optimizer_params={'limit_in_minutes':time_limit_secondes/60},
                      metrics=['accuracy'],
                      best_config_metric='accuracy',
                      inner_cv=KFold(n_splits=3),
                      cache_folder=cache_grid_folder_path,
                      verbosity=0)

pipeline_elements = []

# scaling
pipeline_elements.append(PipelineElement('StandardScaler',  test_disabled=True))
pipeline_elements.append(PipelineElement('Normalizer', test_disabled=True))

prepro_switch = Switch("PreprocSwitch")
prepro_switch += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
prepro_switch += PipelineElement('RandomTreesEmbedding',
                                 hyperparameters={'n_estimators': IntegerRange(10,30),
                                                  'max_depth':IntegerRange(3,6)})
prepro_switch += PipelineElement('SelectPercentile', hyperparameters={'percentile': IntegerRange(5,15)})
#prepro_switch += PipelineElement('FastICA', hyperparameters={'algorithm': Categorical(['parallel', 'deflation'])})


estimator_switch = Switch("EstimatorSwitch")
estimator_switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["linear", "rbf", 'poly', 'sigmoid']),
                                                            'C': FloatRange(0.5, 100),
                                                            'decision_function_shape': Categorical(['ovo', 'ovr']),
                                                            'degree': IntegerRange(2,5)})
estimator_switch += PipelineElement("RandomForestClassifier", hyperparameters={'n_estimators': IntegerRange(10, 100),
                                                                               "min_samples_split": IntegerRange(2, 4)})
estimator_switch += PipelineElement("ExtraTreesClassifier", hyperparameters={'n_estimators':IntegerRange(5,50)})
estimator_switch += PipelineElement("SGDClassifier", hyperparameters={'penalty':Categorical(['l2', 'l1', 'elasticnet'])})

pipeline_elements.append(prepro_switch)
pipeline_elements.append(estimator_switch)

for pipeline_element in pipeline_elements:
    grid_pipe += pipeline_element
    smac_pipe += pipeline_element

grid_pipe.fit(X, y)
smac_pipe.fit(X, y)

y_smac = [1-x.metrics_test[0].value for x in grid_pipe.results.outer_folds[0].tested_config_list]
y_grid = [1-x.metrics_test[0].value for x in smac_pipe.results.outer_folds[0].tested_config_list]

x_smac = list(range(1, len(y_smac)+1))
x_grid = list(range(1, len(y_grid)+1))


y_smac_inc = [min(y_smac[:tmp+1]) for tmp in x_smac]
y_grid_inc = [min(y_grid[:tmp+1]) for tmp in x_grid]

plt.figure(figsize=(10, 7))
plt.plot(x_smac, y_smac, 'b', label='SMAC')
plt.plot(x_smac, y_smac_inc, 'b', label='SMAC Incumbent')
plt.plot(x_grid, y_grid, 'r', label='Grid')
plt.plot(x_grid, y_grid_inc, 'r', label='Grid Incumbent')
plt.title('Optimizer Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
#plt.show()
plt.savefig("optimizer_comparisson.png")

