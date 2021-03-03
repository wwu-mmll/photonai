# How to use custom metrics

 1) You can give PHOTONAI a tuple consisting of a metric name and a function delegate that 
    takes true and predicted values and returns a custom metric
    
 2) You can also use a (custom or existing) class that inherits from keras.metrics.Metric 

```python hl_lines="5 10-16 23-24"
{% include 'examples/basic/custom_metric.py' %}
```