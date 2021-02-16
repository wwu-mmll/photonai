# Switch
The PipelineSwitch element acts like an OR-Operator and decides which element performs best. Currently, you can
only optimize the PipelineSwitch using Grid Search, Random Grid Search and smac3

In this example, we add two different transformer elements and two different estimators, and PHOTONAI will 
evaluate the best choices including the respective hyperparameters.

![PHOTONAI Switch](/assets/images/switch.jpg "PHOTONAI switch pipeline element")


``` python
{% include "basic/switch.py" %}

```