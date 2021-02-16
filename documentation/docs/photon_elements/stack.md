# Stack
You want to do stacking if more than one algorithm shall be applied, which equals to an AND-Operation.

The PHOTONAI Stack delivers the data to all of the entailed PipelineElements and the transformations or 
predictions are afterwards horizontally concatenated.

In this way you can preprocess data in different ways and collect the resulting information to create a new
feature matrix. Additionally, you can train several learning algorithms with the same data in an ensemble-like
fashion and concatenate their predictions to a prediction matrix on which you can apply further processing like
voting strategies.

![PHOTONAI Stack](/assets/images/stack.jpg "PHOTONAI stack pipeline element")

``` python
{% include "basic/stack.py" %}

```