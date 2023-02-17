import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange
from sklearn.datasets import fetch_openml

# blood-transfusion-service-center
blood_transfusion = fetch_openml(name='blood-transfusion-service-center')
X = blood_transfusion.data.values
y = blood_transfusion.target.values
y = (y == '2').astype(int)

my_pipe = Hyperpipe('results_example',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 15, 'acq_func_kwargs': {'kappa': 1}},
                    metrics=['accuracy', 'f1_score'],
                    best_config_metric='f1_score',
                    outer_cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2),
                    inner_cv=StratifiedShuffleSplit(n_splits=4, test_size=0.2),
                    verbosity=0,
                    project_folder='./tmp')

# first normalize all features
my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SVC', hyperparameters={'C': FloatRange(0.1, 150)}, probability=True)

my_pipe.fit(X, y)

# Either, we continue working with the results directly now
handler = my_pipe.results_handler
#, or we load them again later.
# from photonai.processing import ResultsHandler
# handler = ResultsHandler().load_from_file(os.path.join(my_pipe.results.output_folder, "photon_results_file.json"))


# A table with properties and performance of each outer
# fold (and the overall run) is created with the following command.
performance_table = handler.get_performance_table()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(performance_table)
print(" ")

# We now analyze the optimization influence on the result.
config_evals = handler.get_config_evaluations()
for i, j in enumerate(config_evals['f1_score']):
    print("Standard deviation for fold {}: {}.".format(str(i), str(np.std(j))))
print(" ")

# To get an impression of the results,
# it is possible to take a closer look at the test_predictions.
best_config_preds = handler.get_test_predictions()
y_pred = best_config_preds['y_pred']
y_pred_probabilities = best_config_preds['probabilities']
y_true = best_config_preds['y_true']

# While some elements have been misclassified,
# we have a closer look to the elementwise probability.
for i in range(2, 6):
    attribute = "correct" if y_true[i] == y_pred[i] else "incorrect"
    print("Test-element {} was {} predicted "
          "with an assignment probability of {}.".format(str(i), attribute, str(y_pred_probabilities[i])))
