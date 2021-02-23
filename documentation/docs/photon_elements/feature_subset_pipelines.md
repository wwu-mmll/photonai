In PHOTONAI, you can create individual data streams very easily. If, for example, 
you like to apply different preprocessing steps to distinct **subsets of your 
features**, you can create multiple **branches** within your ML pipeline that will hold
any kind of preprocessing. Similarly, you could train different classifiers on different 
feature subsets. 

To add a branch to your pipeline, you can simply create a PHOTONAI Branch and then
add any number of elements to it. If you only add transformer elements to your branch,
the transformed data will be passed to the next element after your branch (or stacked 
in case of a PHOTONAI Stack). If, however, you add a final estimator to your branch, 
the prediction of this estimator will be passed to the next element. You could now
add your created branch to a Hyperpipe, however, creating branches only really makes sense
when having multiple ones and adding those to either a [Stack](stack.md) or 
[Switch](switch.md). Otherwise, why create a branch in the first place? 

Importantly, a branch will always receive **all** of your features if you don't add a
**PHOTONAI DataFilter**. A DataFilter can be added as first element of a branch to make sure
only a specific subset of the features will be passed to the remaining elements of the branch.
It only takes a parameter called `indices` that specifies the data columns that are ultimately
passed to the next element. 

In this example, we create three branches to process three feature subsets of the breast 
cancer dataset separately. For all three branches, we add an SVC to predict the classification
label. This way, PHOTONAI can find the optimal SVC hyperparameter for the three data modalities.
All predictions are then stacked and passed to a final [Switch](switch.md) that will decide between
a Random Forest or another SVC.

``` python
{% include "examples/basic/data_integration.py" %} 

```