# Switch
The [PipelineSwitch element](../../api/base/switch) acts like an OR-Operator and decides which element performs best. 
Currently, you can only optimize the PipelineSwitch using [Grid Search](../../api/optimization/grid_search), 
[Random Grid Search](../../api/optimization/random_grid_search) and [Smac3](../../api/optimization/smac).

In this example, we add two different transformer elements and two different estimators, and PHOTONAI will 
evaluate the best choices including the respective hyperparameters.

![PHOTONAI Switch](https://www.photon-ai.com/static/img/photon/switch.jpg "PHOTONAI switch pipeline element")


``` python
{% include "examples/basic/switch.py" %}

```