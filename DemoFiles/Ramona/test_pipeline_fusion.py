from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineStacking
from sklearn.model_selection import KFold


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


# OPTION 1:
# DEFAULT IS SPLITTING INTO 80% validation and 20% test set
manager = Hyperpipe('outer_man', KFold(n_splits=3), metrics=['accuracy'])

# OPTION 2:
# if you want a specific CV Strategy, use hyperparameter_fitting_cv_object
# test a specific config with KFOld(n_splits=2)
# test the complete hyperparameter search with KFold(n_splits=3)
# manager = Hyperpipe('outer_man', KFold(n_splits=2), metrics=['accuracy'],
#                     hyperparameter_fitting_cv_object=KFold(n_splits=3))

# OPTION 3:
# if you want no splitting at all, use:
# manager = Hyperpipe('outer_man', KFold(n_splits=3), metrics=['accuracy'],
#                    eval_final_performance=False)

svc = PipelineElement.create('svc', {'C': [0.3, 0.5, 1]}, kernel='rbf')
lr = PipelineElement.create('logistic', {'C': [0.7, 0.8]})
fusion = PipelineStacking('estimator_fusion', [svc, lr])
manager.add(fusion)
manager.fit(X, y)

# find the metrics of the TEST SET here:
# manager.test_performances