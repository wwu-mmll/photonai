# Algorithms 

PHOTONAI offers easy access to established machine learning algorithms.

The algorithms can be imported by adding a [PipelineElement](../../api/base/pipeline_element) 
with a specific name, such as _"SVC"_ for importing the [SupportVectorClassifier](
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from 
[scikit-learn](https://scikit-learn.org/stable/), as shown in the following examples.

You can set all parameters of the imported class as usual: e.g. add _gamma='auto'_ to the 
[PipelineElement](../../api/base/pipeline_element) to set the support vector machine's 
gamma parameter to 'auto'. 

In addition, you can specify each parameter as a hyperparameter and define a value range or value list to 
find the optimal value, such as _'kernel': ['linear', 'rbf']_ . 

To build a custom pipeline, have a look at PHOTONAIs [pre-registered processing-](../../algorithms/transformers/) 
and [learning algorithms](../../algorithms/estimators/).
You can access algorithms for all purposes from several open-source packages. In addition, PHOTONAI offers
several utility classes as well, such as linear statistical feature selection or sample pairing algorithms.

In addition you can specify hyperparameters as well as their value range 
in order to be optimized by the hyperparameter optimization strategy. Currently,
PHOTONAI offers [Grid-Search](../../api/optimization/grid_search), [Random Search](
../../api/optimization/random_grid_search) and two frameworks for bayesian optimization.


## PCA
```python
from photonai.base import PipelineElement
PipelineElement('PCA',
                hyperparameters={'n_components': IntegerRange(5, 20)},
                test_disabled=True)
# to test if disabling the PipelineElement improves performance,
# simply add the test_disabled=True parameter
```       
## SVC
```python
PipelineElement('SVC',
                hyperparameters={'kernel': Categorical(['rbf', 'poly']),
                                 'C': FloatRange(0.5, 2)},
                gamma='auto')
```

## Keras Neural Net
```python
PipelineElement('KerasDnnRegressor',
                hyperparameters={'hidden_layer_sizes': Categorical([[10, 8, 4],
                                                                    [20, 5, 3]]),
                                 'dropout_rate': Categorical([[0.5, 0.2, 0.1],
                                                              0.1])},
                activations='relu',
                epochs=5,
                batch_size=32)
```

