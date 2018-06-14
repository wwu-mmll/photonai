![PHOTON LOGO](PhotonLogo.jpg "PHOTON Logo")

# PHOTON
#### A **P**ython-based **H**yperparameter **O**ptimization **To**olbox for **N**eural Networks designed to accelerate and simplify the construction, training, and evaluation of machine learning models.

PHOTON gives you an easy way of setting up a full stack machine learning pipeline including
nested cross-validation and hyperparameter search. After PHOTON has found the best configuration
for your model, it offers a convenient possibility to explore the analyzed hyperparameter space.
It also enables you to persist and load your optimal model, including all preprocessing steps,
with only one line of code.


---

## Table of Contents
[Getting Started](#markdown-header-getting-started)


## Getting Started
In order to use PHOTON you only need to have your favourite Python IDE ready.
Then install it simply via pip
```
pip install photonai
```

## Usage

PHOTON is designed to leave you deciding the important things and automatizing the rest.

The first thing to do is choosing your basic setup:

```python
my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10))
```


- Give your pipeline a **name**.

- Choose a **hyperparameter optimization strategy**.

  Feel free to choose the good old buddy called grid search in order to scan
  the hyperparameter space for the best configuraton. You can also check out his friends
  RandomGridSearch or TimeboxedRandomGridSearch. Add your own optimizer
  by adhering to PHOTON's optimizer interface.

- Which strategies you want to use for the **nested cross-validation**.

  As PHOTON employs nested cross validation you can pick an outer-cross-validation strategy as
  well as an inner-cross-validation strategy. PHOTON expects objects adhering to scikit-learns
  BaseCrossValidator Interface, so you can use any of scikit-learn's
  already implemented cross validation strategies.

- Which **performance metrics** you are interested in

  We registered a lot of performance metrics in PHOTON that you can easily
  pick by its name, such as 'accuracy', 'precision', 'recall', 'f1_score',
  'mean_squared_error', 'mean_absolute_error' etc ..


- Which performance metrics you want to use in order to **pick the best model**

  After the optimization strategy tested a lot of configuration, you tell PHOTON
  which performance metric you want to use in order to pick the best from all
  configurations

### Now you can setup your pipeline elements
As you have customized the generic training and test procedures
with the Hyperpipe construct above, you are now free to add any
preprocessing steps as well as your learning model of choice.

```python
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA', hyperparameters={'n_components': [5, 10, None]}, test_disabled=True)

my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2, "linspace", num=5)})
```

We are using a very basic setup here:
1. normalize all features using a standard scaler
2. do feature selection using a PCA
3. learn the task good old SVM for Classification


For each element you can specifiy which hyperparameters to test, that is
you declare the hyperparameter space in which the optimizer looks for the
optimum configuration.

You can give PHOTON a list of distinct values of any kind, or use
PHOTON classes for specifying the hyperparamter space:
- Categorical
- FloatRange (can generate a range, a linspace or a logspace)
- IntegerRange (can generate a range, a linspace or a logspace)
- BooleanSwitch (try both TRUE and FALSE)

### TRAIN AND TEST THE MODEL

You start the training and test procedure including the hyperparamter search
by simply asking the hyperpipe to fit to the data and targets.

```python
my_pipe.fit(X, y)
```


### SAVE BEST PERFORMING MODEL

After the training and testing is done and PHOTON found the optimum
configuation it automatically fits your pipeline to that best configuration.
You can save the complete pipeline for further use.

```python
my_pipe.save_optimum_pipe('/home/photon_user/photon_test/optimum_pipe.photon')
```


### EXPLORE HYPERPARAMETER SEARCH RESULTS

The PHOTON Investigator is a convenient web-based tool for analyzing the training
and test perofrmances of the configurations explored in the hyperparamter search.

You start it by calling
```python
Investigator.show(my_pipe)
```



## COMPLETE EXAMPLE

```python
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.optimization.SpeedHacks import MinimumPerformance
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister

from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
# mongo_settings = PersistOptions(mongodb_connect_url="mongodb://localhost:27017/photon_db",
#                                 save_predictions=False,
#                                 save_feature_importances=False)


save_options = PersistOptions(local_file="/home/photon_user/photon_test/test_item.p")


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe_no_performance',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy', 'precision', 'recall'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=10),  # test each configuration ten times respectively
                    # skips next folds of inner cv if accuracy and precision in first fold are below 0.96.
                    performance_constraints=[MinimumPerformance('accuracy', 0.96),
                                             MinimumPerformance('precision', 0.96)],
                    verbosity=1, # get error, warn and info messages
                    persist_options=save_options)


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
PhotonRegister.list()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
PhotonRegister.info('SVC')


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': [5, 10, None]}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# AND SHOW THE RESULTS IN THE WEBBASED PHOTON INVESTIGATOR TOOL
Investigator.show(my_pipe)

# YOU CAN ALSO SAVE THE BEST PERFORMING PIPELINE FOR FURTHER USE
my_pipe.save_optimum_pipe('/home/photon_user/photon_test/optimum_pipe.photon')

# YOU CAN ALSO LOAD YOUR RESULTS FROM THE MONGO DB
# Investigator.load_from_db(mongo_settings.mongodb_connect_url, my_pipe.name)
```