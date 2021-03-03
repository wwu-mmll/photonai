# Stack
You want to do stacking if more than one algorithm shall be applied, which equals to an AND-Operation.

The [PHOTONAI Stack](../../api/base/stack) delivers the data to all of the entailed [PipelineElements](
../../api/base/pipeline_element) and the transformations or predictions are afterwards horizontally concatenated.

In this way you can preprocess data in different ways and collect the resulting information to create a new
feature matrix. Additionally, you can train several learning algorithms with the same data in an ensemble-like
fashion and concatenate their predictions to a prediction matrix on which you can apply further processing like
voting strategies.

![PHOTONAI Stack](https://www.photon-ai.com/static/img/photon/stack.jpg "PHOTONAI stack pipeline element")

``` python
{% include "examples/basic/stack.py" %}

```