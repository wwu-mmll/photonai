<h1>Callback Elements</h1>
PHOTONAI implements pipeline callbacks which allow for live inspection of the data flowing through the 
pipeline at runtime. _Callbacks_ act as pipeline elements and can be inserted at any point within the pipeline. 
They must define a function delegate which is called with the exact same data that the next pipeline step will receive. 

Thereby, a developer may inspect e.g. the shape and values of the feature matrix after a sequence of 
transformations have been applied. Return values from the delegate functions are ignored, 
so that after returning from the delegate call, the original data is directly passed to the next processing step.

```python hl_lines="7-8 32"
{% include 'examples/advanced/callbacks.py' %}
```