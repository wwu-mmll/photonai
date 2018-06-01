"""
Test Feature Selection
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from Framework.PhotonBase import Hyperpipe, PipelineElement

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# create cross-validation object first
cv_object = KFold(n_splits=3, shuffle=True, random_state=0)

# create a hyperPipe
manager = Hyperpipe('god', cv_object, optimizer='random_grid_search')

manager += PipelineElement.create('f_classif_select_percentile',
                                  {'percentile': [10,20,30,100]},test_disabled=True)

# SVMs (linear and rbf)
manager += PipelineElement.create('svc', {}, kernel='linear')

manager.fit(X, y)