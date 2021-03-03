# How and why to apply preprocessing

There are transformations that can be applied to the data as a whole BEFORE it is split into different 
training, validation, and testing subsets. Thereby, computational resources are saved as these operations
are not repeated for each of the cross-validation folds. Importantly, this does only make sense for transformations
that do not fit a model to the dataset as a group (such as e.g. a PCA) but rather apply transformations on a single-
subject level (such as resampling a nifti image). 

```python hl_lines="21-24"
{% include "examples/basic/preprocessing.py" %}
```

