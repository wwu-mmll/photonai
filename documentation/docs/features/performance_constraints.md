<h1>Performance Constraints</h1>
<div class="photon-docu-header">
    <p>
        Integrating performance baselines and performance expectations in the hyperparameter optimization process is
        furthermore helpful to increase the overall speed and efficiency. The user can specify to skip the testing of
        a specific hyperparameter configuration in further inner-cross-validation folds if the given configuration
        performs worse than a given static or dynamic threshold.
    </p>
</div>

``` python
{% include "advanced/regression_with_constraints.py" %}

```