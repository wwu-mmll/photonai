import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
from sklearn import linear_model, decomposition, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from six.moves import cPickle as pickle

from PipelineWrapper.TFDNNClassifier import TFDNNClassifier


# setup pipeline
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
dnn = TFDNNClassifier(0.5)

# ('logistic', logistic),
pipe = Pipeline(steps=[('pca', pca),  ('dnn', dnn)])

data_root = '/home/rleenings/PycharmProjects/TFLearnTest/'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
all_data = pickle.load(open(pickle_file, "rb"))

train_data = all_data['train_dataset']
train_labels = all_data['train_labels']
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

test_data = all_data['test_dataset']
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
test_labels = all_data['test_labels']

# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target

# plt.figure()
# plt.imshow(X_digits[5, :].reshape(8, 8))
# plt.show()

X_digits = train_data[1:2000, :]
y_digits = train_labels[1:2000]

###############################################################################
# Plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')


###############################################################################
# Prediction

#Cs = np.logspace(-4, 4, 5)
# ,logistic__C=Cs))

n_components = [60, 80, 100, 120]
grad_desc_alphas = [0.1, 0.3, 0.5]

# Parameters of pipelines can be set using ‘__’ separated parameter names:
gs_params = dict(pca__n_components=n_components, dnn__gd_alpha=grad_desc_alphas)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe.steps])
print("parameters:")
pprint(gs_params)
t0 = time()

grid_search = GridSearchCV(pipe, gs_params) #, n_jobs=-1, verbose=1)

grid_search.fit(X_digits, y_digits)

print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(gs_params.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


y_test = grid_search.predict(test_data)
print(classification_report(y_test, test_labels))

print("Best gradient descent: %.0.3f" % grid_search.best_estimator_.named_steps['dnn'].gd_alpha)

plt.axvline(grid_search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()