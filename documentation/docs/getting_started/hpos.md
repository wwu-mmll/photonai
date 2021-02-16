<h1>Hyperparameter Optimization</h1>

PHOTONAI offers easy access to several established Hyperparameter Optimization Strategies.

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
                 optimizer_params={'n_configurations': 30})
```
<h3>Timeboxed Random Grid Search</h3>
<h3>Random Search</h3>
<h3>Scikit-Optimize</h3>
<h3>Nevergrad</h3>
<h3>Smac</h3>
<h3>Switch Optimizer</h3>
This optimizer is special, as it uses the strategies above to optimizes the same dataflow for different 
learning algorithms in a switch ("OR") element at the end of the pipeline. 

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