In some situations, and especially in the life sciences, we are interested in 
the predictive value of certain features but would therefore like to exclude
the contribution of confounding variables. In order to do that, simple linear models
can be used to regress out the effect of a confounder from all of the features.
However, to ensure the independence of training and test set, this **has** to be done
within the cross-validation framework. Adding a ConfoundRemoval PipelineElement
to a PHOTONAI pipeline will ensure exactly that when regressing out confounding effects.
The confounder variables can be passed to the Hyperpipe in the .fit() method (see 
example below).

``` python hl_lines="32-35 40"
{% include "examples/advanced/confounder_removal_example.py" %} 

```