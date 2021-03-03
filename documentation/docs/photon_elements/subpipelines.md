<h1>Subpipelines</h1>
<div class="photon-docu-header">
    <p>
        If the user wants to parallelize a complete sequence of transformations, that is not only singular
        PipelineElements but an ordered number of PipelineElements, the class PHOTONAI Branch offers a way to create
        parallel subpipelines.
    </p>
    <p>
        The branch in turn, can be used in combination with the AND- and OR- Elements in order to design complex
        pipeline architectures.
    </p>
</div>

``` python
{% include "examples/advanced/pipeline_branches.py" %} 

```