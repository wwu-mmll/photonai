<h1>Hyperparameter Optimization</h1>

PHOTONAI offers easy access to several established hyperparameter optimization strategies.

<h3>Grid Search</h3>
An exhaustive searching through a manually specified subset of the hyperparameter space. The grid is defined by 
a finite list for each hyperparameter. 

```python
pipe = Hyperpipe("...", 
                 optimizer='grid_search')
```
<h3>Random Grid Search</h3>
Random sampling of a manually specified subset of the hyperparameter space. The grid is defined by 
a finite list for each hyperparameter. Then, a specified number of random configurations from this grid is tested

```python
pipe = Hyperpipe("...", 
                 optimizer='random_grid_search',
                 optimizer_params={'n_configurations': 30,
                                   'limit_in_minutes': 10})
```

<h3>Random Search</h3>
A grid-free selection of configurations based on the hyperparameter space. In the case of numerical parameters, 
decisions are made only on the basis of the interval limits. The creation of configurations is limited 
by time or a maximum number of runs.
```python
pipe = Hyperpipe("...", 
                 optimizer='random_search',
                 optimizer_params={'n_configurations': 30,
                                   'limit_in_minutes': 20})
```

<h3>Scikit-Optimize</h3>
Scikit-Optimize, or skopt, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions.
It implements several methods for sequential model-based optimization.
skopt aims to be accessible and easy to use in many contexts.

Scikit-optimize usage and implementation details available [here](https://scikit-optimize.github.io/stable/).
A detailed parameter documentation [here.](
    https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer)
```python
pipe = Hyperpipe("...", 
                 optimizer='sk_opt',
                 optimizer_params={'n_configurations': 55,
                                   'n_initial_points': 15,
                                   'initial_point_generator': "sobol",
                                   'acq_func': 'LCB',
                                   'acq_func_kwargs': {'kappa': 1.96}})
```
<h3>Nevergrad</h3>
Nevergrad is a gradient-free optimization platform. 
Thus, this package is suitable for optimizing over the hyperparamter space.
As a great advantage, evolutionary algorithms are implemented here 
in addition to Bayesian techniques.

Nevergrad usage and implementation details available [here](
https://facebookresearch.github.io/nevergrad/).
```python
import nevergrad as ng
# list of all available nevergrad optimizer
print(list(ng.optimizers.registry.values()))
my_pipe = Hyperpipe("...", 
                    optimizer='nevergrad',
                    optimizer_params={'facade': 'NGO', 'n_configurations': 30})
```

<h3>Smac</h3>

SMAC (sequential model-based algorithm configuration) is a
versatile tool for optimizing algorithm parameters.
The main core consists of Bayesian Optimization in
combination with an aggressive racing mechanism to efficiently
decide which of two configurations performs better.

SMAC usage and implementation details available [here](
    https://automl.github.io/SMAC3/master/quickstart.html).

```python
my_pipe = Hyperpipe("...",
                    optimizer='smac',
                    optimizer_params={"facade": "SMAC4BO",
                                      "wallclock_limit": 60.0*10,  # seconds
                                      "ta_run_limit": 100}  # limit of configurations
                    )
```


<h3>Switch Optimizer</h3>
This optimizer is special, as it uses the strategies above to optimizes the same dataflow for different 
learning algorithms in a [switch ("OR") element](../../api/base/switch) at the end of the pipeline. 

For example you can use bayesian optimization for each learning algorithm and select that each of the algorithms
gets 25 configurations to be tested. 

This is different to a global optimization, in which, after an initial exploration phase, computational resources 
are dedicated to the best performing learning algorithm only. 

By equally distributing computational ressources to each learning algorithms, better comparability is achieved 
in-between the algorithms. This can according to the use case be desirable.  
```python
pipe = Hyperpipe("...",
                 optimizer="switch",
                 optimizer_params={'name': 'sk_opt', 'n_configurations': 25})
```