from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from Framework.PhotonBase import Hyperpipe, PipelineElement

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)


manager = Hyperpipe('god', inner_cv, optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1})
svc_estimator = PipelineElement.create('svc', {'kernel': ['linear', 'rbf']})
manager.add(svc_estimator)

# manager.fit(X, y)
# manager.optimum_pipe.fit(X, y)
# new_y = manager.optimum_pipe.predict(X)

preds = []

for train, test in outer_cv.split(X):
    print(manager)
    manager.fit(X[train], y[train])
    manager.optimum_pipe.fit(X[train], y[train])
    y_pred_test = manager.optimum_pipe.predict(X[test])
    print(len(y_pred_test))
    preds.append(y_pred_test)


# example = ('sklearn.decomposition', 'PCA')
# item = __import__(example[0], globals(), locals(), example[1], 0)
#
# base_element = getattr(item, example[1])
# desired_instance = base_element()

# import itertools
#pipeline_items = {'1': 1, '2': 1, '3': 0, '4': 0, '5': 1, '6': 0, '7': 0}
# pipeline_items = [1, 1, 0, 0, 1, 0, 0]
# items_to_disable = [i for i, x in enumerate(pipeline_items) if x]
# print(items_to_disable)
#
# lst = [list(i) for i in itertools.product([0, 1], repeat=len(items_to_disable))]
#
# A = [{'A': 1}, {'A': 2}]
# B = [{'B': 1}, {'B': 2}]
# C = [{'C': 1}, {'C': 2}]
#
# D = [A, B, C]
# DE = [sublist for sublist in D]
# prod = itertools.product(*DE)
# for p in prod:
#     print(p)
#
