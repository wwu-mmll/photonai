from sklearn.datasets import load_breast_cancer, load_boston
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

import unittest
from photonai.modelwrapper.keras_base_models import KerasBaseClassifier, KerasBaseRegressor


class KerasBaseClassifierTest(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        self.model = Sequential()
        self.model.add(Dense(3, input_dim=self.X.shape[1], activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.estimator_type = KerasBaseClassifier

    def test_predict(self):
        estimator = self.estimator_type(model=self.model,
                                        epochs=10,
                                        nn_batch_size=2,
                                        verbosity=0
                                        )
        estimator.fit(self.X, self.y)
        res = estimator.predict(self.X)
        self.assertTrue(isinstance(res, np.ndarray))


class KerasBaseRegressorTest(KerasBaseClassifierTest):

    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)

        self.model = Sequential()
        self.model.add(Dense(5, input_dim=self.X.shape[1], activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        self.estimator_type = KerasBaseRegressor
