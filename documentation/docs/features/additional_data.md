# Stream and Access Additional Data
Numerous use-cases rely _on data not contained in the feature matrix_ at runtime, e.g. when aiming to control for the 
effect of covariates. In PHOTONAI, additional data can be streamed through the pipeline and is accessible for 
all pipeline steps while - importantly - being matched to the (nested) cross-validation splits.

```python
{% include 'examples/advanced/additional_data.py' %}
```