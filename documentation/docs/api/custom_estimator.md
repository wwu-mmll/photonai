<h1>Custom Estimator</h1>
<div class="photon-docu-header">
  <p>
      You can combine your own learning algorithm, bet it a neural net or anything else, by simply adhering to
      the <a href="http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects" target="_blank">scikit-learn interface</a> as shown below.
      Then register your class with the Register module and you're done!
  </p>
</div>

``` python
{% include "examples/advanced/custom_elements/custom_estimator.py" %} 

```