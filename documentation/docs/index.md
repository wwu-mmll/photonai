<h1>Getting Started</h1>
<h2>Installation</h2>
<p class="small-p">You only need two things: Python 3 and your favourite Python IDE to get started. Then simply install via pip.</p>

```python
pip install photonai
```

<h2> Setup Default Analysis </h2>
<h3>Regression with default pipeline</h3>

```python
from sklearn.datasets import load_diabetes
from photonai import RegressionPipe

my_pipe = RegressionPipe('diabetes')
X, y = load_diabetes(return_X_y=True)
my_pipe.fit(X, y)
```

<h3>Classification with modified default pipeline</h2>
``` python
from photonai import ClassificationPipe
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit

X, y = load_breast_cancer(return_X_y=True)
my_pipe = ClassificationPipe(name='breast_cancer_analysis',
                             inner_cv=ShuffleSplit(n_splits=2),
                             scaling=True,
                             imputation=False,
                             imputation_nan_value=None,
                             feature_selection=False,
                             dim_reduction=True,
                             n_pca_components=10)
my_pipe.fit(X, y)
```


<h2>Or Setup New Custom Analysis</h2>

Start by importing some utilities. and creating a new Hyperpipe instance, naming the analysis and specifying where to save all outputs.
Select parameters to customize the training, hyperparameter optimization and testing procedure.
Particularly, you can choose the hyperparameter optimization strategy, set parameters, choose performance metrics
and choose the performance metric to minimize or maximize, respectively.
    
```python
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.datasets import load_breast_cancer
from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange, FloatRange

pipe = Hyperpipe('basic_pipe', project_folder='./',

                  # choose hyperparameter optimization strategy
                  optimizer='random_grid_search',

                  # PHOTONAI automatically calculates your preferred metrics
                  metrics=['accuracy', 'balanced_accuracy', 'f1_score'],

                  # this metrics selects the best hyperparameter configuration
                  # in this case mean squared error is minimized
                  best_config_metric='f1_score',

                  # select cross validation strategies
                  outer_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                  inner_cv=KFold(n_splits=10))
``` 

<h3>Add pipeline elements to your liking.</h3>
Select and arrange normalization, dimensionality reduction, feature selection, data augmentation,
over- or undersampling algorithms in simple or parallel data streams. You can integrate
custom algorithms or choose from our wide range of pre-registered algorithms from established toolboxes.

```python
pipe += PipelineElement('StandardScaler')

pipe += PipelineElement('PCA',
                        hyperparameters={'n_components': FloatRange(0.5, 0.8, step=0.1)})

pipe += PipelineElement('ImbalancedDataTransformer',
                        hyperparameters={'method_name': ['RandomUnderSampler',
                                                         'RandomOverSampler',
                                                         'SMOTE']})

or_element = Switch('EstimatorSwitch')
or_element += PipelineElement('RandomForestClassifier',
                              hyperparameters={'min_samples_split': IntegerRange(2, 30)})
or_element += PipelineElement('SVC',
                              hyperparameters={'C': FloatRange(0.5, 10),
                                               'kernel': ['linear', 'rbf']})

pipe += or_element
```

<h3>Finally, load data and start training!</h3> 
Load your data and start the (nested-) cross-validated hyperparameter optimization, training and evaluation procedure.
You will see an extensive output to monitor the hyperparameter optimization progress, see the results and track the
best performances so far.


```python
X, y = load_breast_cancer(return_X_y=True)
pipe.fit(X, y)
```



