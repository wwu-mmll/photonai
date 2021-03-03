# Comparing Estimators

With the specialized switch optimizer the user can allocate 
the same computational resource to hyperparameter optimize the pipeline for each learning algorithm
in a final [switch element](../../api/base/switch), respectively.

The user chooses a hyperparameter optimization strategy, to be applied to optimize the pipeline for each learning 
algorithm in a distinct hyperparameter space. Thereby each algorithm is optimized with the pipeline 
with the same settings, so that comparability between the learning algorithms is given. 

Another strategy would be to optimize estimator selection within a unified hyperparameter space, e.g. by applying the
[smac3 optimizer](../../api/optimization/smac). Within a unified hyperparameter space there is an exploration phase,
after which only the most promising algorithms receive further computational time and 
thus, some learning algorithms receive more computational resources than others. This strategy is capable to auto-
matically select the best algorithm, however it is due to the given reasons less suitable for algorithm comparisons. 

With the last line of code in this example, the user requests a comparative performance metrics table, that 
shows the mean _validation_ performances for the best configurations found in each outer fold for each estimator,
respectively. 
```python hl_lines="10 11 35 24 40"
{% include 'examples/optimizer/meta_optimizer.py' %}

```