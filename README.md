![PHOTON LOGO](http://www.photon-ai.com/static/img/photon/photon-logo-github.png "PHOTON Logo")

[![Build Status](https://travis-ci.com/wwu-mmll/photonai.svg?branch=master)](https://travis-ci.com/wwu-mmll/photonai)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/photonai/badge.svg?branch=master)](https://coveralls.io/github/wwu-mmll/photonai?branch=master)
[![Github Contributors](https://img.shields.io/github/contributors-anon/wwu-mmll/photonai?color=blue)](https://github.com/wwu-mmll/photonai/graphs/contributors)
[![Github Commits](https://img.shields.io/github/commit-activity/y/wwu-mmll/photonai)](https://github.com/wwu-mmll/photonai/commits/master)
[![PyPI Version](https://img.shields.io/pypi/v/photonai?color=brightgreen)](https://pypi.org/project/photonai/)
[![License](https://img.shields.io/github/license/wwu-mmll/photonai)](https://github.com/wwu-mmll/photonai/blob/master/LICENSE)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fwwu_mmll)](https://twitter.com/wwu_mmll)

#### PHOTONAI is a high level python API for designing and optimizing machine learning pipelines.

We create a system in which you can easily select and combine both pre-processing and learning algorithms from
state-of-the-art machine learning toolboxes,
 and arrange them in simple or parallel pipeline data streams. 
 
 In addition, you can parametrize your training and testing
 workflow choosing cross-validation schemas, performance metrics and hyperparameter
 optimization metrics from a list of pre-registered options. 
 
 Importantly, you can integrate custom solutions into your data processing pipeline, 
 but also for any part of the model training and evaluation process including custom
 hyperparameter optimization strategies.  

For a detailed description, 
__[visit our website and read the documentation](https://www.photon-ai.com)__

or you can read a prolonged introduction on [Arxiv](https://arxiv.org/abs/2002.05426)



---
## Getting Started
In order to use PHOTONAI you only need to have your favourite Python IDE ready.
Then install the latest stable version simply via pip
```
pip install photonai
# Or try out the latest features if you don't rely on a stable version, using:
pip install --upgrade git+https://github.com/wwu-mmll/photonai.git@develop
```

You can setup a full stack machine learning pipeline in a few lines of code:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical, IntegerRange

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    # which optimizer PHOTONAI shall use
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    # the performance metrics of your interest
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    # after hyperparameter optimization, this metric declares the winner config
                    best_config_metric='accuracy',
                    # repeat hyperparameter optimization three times
                    outer_cv=KFold(n_splits=3),
                    # test each configuration five times respectively,
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    project_folder='./tmp/')


# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))

# then do feature selection using a PCA
my_pipe += PipelineElement('PCA', 
                           hyperparameters={'n_components': IntegerRange(5, 20)}, 
                           test_disabled=True)

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', 
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(0.5, 2)}, gamma='scale')

# train pipeline
X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)
```
---
## Features

#### Easy access to established ML implementations
We pre-registered diverse preprocessing and learning algorithms from 
state-of-the-art toolboxes e.g. scikit-learn, keras and imbalanced learn to 
rapidly build custom pipelines

#### Hyperparameter Optimization
With PHOTONAI you can seamlessly switch between diverse hyperparameter 
optimization strategies, such as (random) grid-search
 or bayesian optimization (scikit-optimize, smac3).

#### Extended ML Pipeline
You can build custom sequences of processing and learning algorithms with a simple syntax. 
PHOTONAI offers extended pipeline functionality such as parallel sequences, custom callbacks in-between pipeline 
elements, AND- and OR- Operations, as well as the possibility to flexibly position data augmentation, class balancing
or learning algorithms anywhere in the pipeline.

#### Model Sharing
PHOTONAI provides a standardized format for sharing and loading optimized pipelines across 
platforms with only one line of code.

#### Automation
While you concentrate on selecting appropriate processing steps, learning algorithms, hyperparameters and
training parameters, PHOTONAI automates the nested cross-validated optimization and evaluation loop for any custom pipeline.

#### Results Visualization
PHOTONAI comes with extensive logging of all information in the training, testing and hyperparameter 
optimization process. In addition, optimum performances and the hyperparameter optimization progress 
are visualized in the [PHOTONAI Explorer](https://explorer.photon-ai.com).

#### For more use cases, examples, contribution guidelines and API details visit our website
## [www.photon-ai.com](https://www.photon-ai.com)  
