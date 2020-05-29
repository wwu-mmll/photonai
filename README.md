![PHOTON LOGO](http://www.photon-ai.com/static/img/PhotonLogo.jpg "PHOTON Logo")

#### PHOTON is a high level python API for designing and optimizing machine learning pipelines.

We developed a framework which pre-structures and automatizes the repetitive part of the model development process so that the user can focus on the important design decisions regarding pipeline architecture and the choice of parameters.

By treating each pipeline element as a building block, we create a system in which the user can select and combine processing steps, adapt their arrangement or stack them in more advanced pipeline layouts.

PHOTON is designed to give any user easy access to state-of-the-art machine learning and integrates the power of various machine learning toolboxes and algorithms.

### [Read our mission statement on Arxiv](https://arxiv.org/abs/2002.05426)

[Read the Documentation](https://www.photon-ai.com)

---
## Getting Started
In order to use PHOTON you only need to have your favourite Python IDE ready.
Then install it simply via pip
```
pip install git+https://github.com/photon-team/photon
```

You can setup a full stack machine learning pipeline in a few lines of code:

```python
imports ...

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    # which optimizer PHOTON shall use
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 10},
                    # the performance metrics of your interest
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    # after hyperparameter optimization, this metric declares the winner config
                    best_config_metric='accuracy',
                    # repeat hyperparameter optimization three times
                    outer_cv=KFold(n_splits=3),
                    # test each configuration five times respectively,
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    output_settings=OutputSettings(project_folder='./tmp/'))


# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))

# then do feature selection using a PCA
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2)}, gamma='scale')

# train pipeline
X, y = load_breast_cancer(True)
my_pipe.fit(X, y)

# visualize results
Investigator.show(my_pipe)
```
---
## Features

**PHOTON is designed to give all the power to the important decicions in the model development process.**

### Automatized Training and Test Procedure
- It offers prestructured and automatized training and test procedure integrating nested cross-validation and hyperparameter optimization
- includes hyperparameter optimization strategies (grid_search, random_grid_search, scikit-optimize, smac3)
and an easy way to specify hyperparameters
- automatically computes the performance metrics of your interest
- automatically chooses the best hyperparameter configuration 
- standardized format for saving, loading and distributing optimized and fully trained pipeline architectures with only one line of code
- persist hyperparameter optimization process logs in local file or MongoDB

[(see Hyperpipe)](http://www.photon-ai.com/documentation/hyperpipe)


### Convenient Functionality for Designing ML Pipeline Architectures 
-  provides access to manifold algorithms from diverse machine learning python toolboxes, that can be used without the 
need to acquire toolbox specific syntax skills 
[(see PipelineElement)](http://www.photon-ai.com/documentation/pipeline_element)
- compatible with all scikit-learn algorithms 
[(see Classifier Ensemble)](http://www.photon-ai.com/documentation/classifier_ensemble)
- compatible with kera, tensorflow and other python neural net model development libraries
[(see Keras DNN Multiclass Classifier)](http://www.photon-ai.com/documentation/keras_multiclass)
- integrate custom neural net models
[(see Custom Neural Net)](http://photon-ai.com/documentation/neural_net) 
- add any custom learning algorithm 
[(see Custom Estimator)](http://www.photon-ai.com/documentation/)
- add any custom preprocessing method 
[(see Custom Transformer)](http://www.photon-ai.com/documentation/)
- handles parallel data streams encapsulated in AND-Elements
[(see PHOTON Stack)](http://www.photon-ai.com/documentation/stack_element)
- offers automatic selection of two or more competing algorithms in OR-Elements 
[(see PHOTON Switch)](http://www.photon-ai.com/documentation/switch_element) 
- has the possibility to branch of several parallel sub-pipelines each containing a sequence of data transformations.
[(see PHOTON Branch)](http://www.photon-ai.com/documentation/subpipelines)
- automatizes statistical validtion using PermutationTests
[(see PermutationTest)](http://www.photon-ai.com/documentation/permutation_test)

### Integration of state-of-the art algorithms in the field 
- handles dynamic target manipulations, e.g. for Data Augmentation
[(see Sample Paring)](http://www.photon-ai.com/documentation/sample_pairing) 
- integrates functionality for imbalanced datasets from imblearn for Over- and Undersampling 
[(see Over- and Undersampling)](http://www.photon-ai.com/documentation/imbalanced_data)  
- provides access to supplementary data not included in the feature matrix matched to the cross-validation split,
e.g. for confounder removal [(see Confounder Removal)](http://www.photon-ai.com/documentation/confounder_removal)
- provides a module for model development specialized on neuroimaging data [(see Brain Age)](http://www.photon-ai.com/documentation/brain_age) 

### Visualization of model performance and the hyperparameter optimization process
- web-based investigation tool based on microframework Flask
- convenient exploration of the hyperparameter search results and performance visualization
[(see Investigator)](http://www.photon-ai.com/documentation/investigator)

###[Read the Documentation and explore the Examples](https://www.photon-ai.com/documentation)
