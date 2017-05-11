from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineFusion
from sklearn.model_selection import KFold


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


# OPTION 1:
# DEFAULT IS SPLITTING INTO 80% validation and 20% test set
outer_man = Hyperpipe('outer_man', KFold(n_splits=2), metrics=['accuracy'])

# OPTION 2:
# if you want a specific CV Strategy, use hyperparameter_fitting_cv_object
# test a speficic config with KFOld(n_splits=2)
# test the complete hyperparameter search with KFold(n_splits=3)
# outer_man = Hyperpipe('outer_man', KFold(n_splits=2), metrics=['accuracy'],
#                     hyperparameter_fitting_cv_object=KFold(n_splits=3))

# OPTION 3:
# if you want no splitting at all, use:
# outer_man = Hyperpipe('outer_man', KFold(n_splits=3), metrics=['accuracy'],
#                    eval_final_performance=False)

outer_man.add(PipelineElement.create('test_wrapper', {'any_param': [1, 2]}))

# create a second level pipe
inner_man = Hyperpipe('inner_man', KFold(n_splits=3), local_search=True, metrics=['accuracy'],
                      hyperparameter_search_cv_object=KFold(n_splits=2))
inner_man.add(PipelineElement.create('svc', {'C': [0.3, 0.5, 1]}, kernel='rbf'))

pipeline_fusion = PipelineFusion('fusion_element', [inner_man])

# add inner to outer pipeline with the help of the pipeline fusion wrapper
outer_man.add(pipeline_fusion)
outer_man.fit(X, y)
