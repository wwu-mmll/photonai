import numpy as np
from photonai.validation.cross_validation import OutlierKFold

X = np.random.randint(10, size=(100, 20))
y_one = np.ones((80,))
y_minus_one = np.ones((20,)) * -1
y = np.concatenate((y_one, y_minus_one))

outer_k = OutlierKFold()
train_list = list(outer_k.split(X, y))


debug = True
