<h1>Stack</h1>
<div class="photon-docu-header">
    <p>
        You want to do stacking if more than one algorithm shall be applied, which equals to an AND-Operation.
    </p>
    <p>
        The PHOTONAI Stack delivers the data to all of the entailed PipelineElements and the transformations or 
        predictions are afterwards horizontally concatenated.
    </p>
    <p>
        In this way you can preprocess data in different ways and collect the resulting information to create a new
        feature matrix. Additionally, you can train several learning algorithms with the same data in an ensemble-like
        fashion and concatenate their predictions to a prediction matrix on which you can apply further processing like
        voting strategies.
    </p>
</div>

``` python
{% include "basic/stack.py" %}

```