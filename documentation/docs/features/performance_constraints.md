# Performance Constraints

Integrating performance baselines and performance expectations in the hyperparameter optimization process is
furthermore helpful to increase the overall speed and efficiency. Further testing of
a specific hyperparameter configuration in further inner-cross-validation folds can be skipped 
if the given configuration performs worse than a given static or dynamic threshold.

There are three types of contraints implemented in PHOTONAI:

 * _MinimumPerformanceConstraint_: the lower bound is the given threshold (e.g. accuracy of at least 0.8)
 * _BestPerformanceConstraint_: the lower bound (+- margin) is the so far best metric value 
 * _DummyPerformanceConstraint_: the lower bound (+-margin) is the dummy performance of the specific metric

The threshold is applied in three strategies:

 * _any_: Computation is skipped if any of the folds is worse than the threshold
 * _first_: Computation is skipped if the first fold performs worse than the threshold
 * _mean_: Computation is skipped if the mean of all folds computed so far is worse than the threshold

``` python hl_lines="19-21"
{% include "examples/advanced/regression_with_constraints.py" %}

```