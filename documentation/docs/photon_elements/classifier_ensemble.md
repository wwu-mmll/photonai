An ensemble is a combination of multiple base estimators. For a short introduction to ensemble
methods, see [Sklearn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html). 
In PHOTONAI, an estimator ensemble can be easily created by adding any number of estimators
to a [Stack](stack.md). Afterwards, simply add a meta estimator that receives the predictions of
your stack. This can be any estimator or simply a averaging or voting strategy. In this example,
we used the PhotonVotingClassifier to create a final prediction.
``` python
{% include "examples/basic/classifier_ensemble.py" %} 

```