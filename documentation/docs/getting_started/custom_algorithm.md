<h1>Add a custom algorithm</h1>

In order to integrate a custom algorithm in PHOTONAI, all you need to do is provide a class adhering to the popular
[scikit-learn object API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html). 
In the following we will demonstrate an example to integrate a custom transformer to the 
[Hyperpipe](../../api/base/hyperpipe). 

First, implement your data processing logic like this.
```python
{% include 'examples//advanced/custom_elements/custom_transformer.py' %}
```

Afterwards, register your element with the photon registry like this. 
Custom elements must only be registered once.

```python
from photonai.base import PhotonRegistry

custom_element_root_folder = "./"
registry = PhotonRegistry(custom_elements_folder=custom_element_root_folder)

registry.register(photon_name='MyCustomTransformer',
                  class_str='custom_transformer.CustomTransformer',
                  element_type='Transformer')

# show information about the element
registry.info("MyCustomTransformer")
```
Afterwards, you can use your custom element in the pipeline like this. 
Importantly, the custom_elements_folder must be activated for each use as the folder's content, and therefore the
custom class implementation might otherwise not be accessible by the python script. 

```python
from photonai.base import PhotonRegistry, Hyperpipe, PipelineElement

custom_element_root_folder = "./"
registry = PhotonRegistry(custom_elements_folder=custom_element_root_folder)

# This add the custom algorithm folder to the python path in order to import and instantiate the algorithm 
registry.activate()

# then use it 
my_pipe = Hyperpipe("...")
my_pipe += PipelineElement('MyCustomTransformer', hyperparameters={'param1': [1, 2, 3]})
```