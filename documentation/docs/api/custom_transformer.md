<h1>Custom Transformer</h1>
<div class="photon-docu-header">
  <p>
      You can add your own method, be it preprocessing, feature selection or dimensionality reduction, by simply adhering to
      the <a href="http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects" target="_blank">scikit-learn interface</a> as shown below.
      Then register your class with the Register module and you're good to go. You can then combine it with any optimizer and metric and design your custom pipeline layout.
  </p>
</div>

``` python
{% include "examples/advanced/custom_elements/custom_transformer.py" %} 

```