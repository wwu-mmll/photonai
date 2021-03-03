from sklearn.datasets import load_breast_cancer, load_boston
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import numpy as np
import warnings
import os

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

        inputs = tf.keras.Input(shape=(self.X.shape[1],))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)
        self.tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.tf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def test_predict(self):
        estimator = self.estimator_type(model=self.model,
                                        epochs=3,
                                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
                                        nn_batch_size=2,
                                        verbosity=0
                                        )
        estimator.fit(self.X, self.y)
        res = estimator.predict(self.X)
        self.assertTrue(isinstance(res, np.ndarray))

        # small data input
        with warnings.catch_warnings(record=True) as w:
            estimator.fit(self.X[:99, :], self.y[:99])
            res = estimator.predict(self.X[:20])
            assert any("Cannot use validation split because of small" in s for s in [e.message.args[0] for e in w])
            self.assertTrue(isinstance(res, np.ndarray))

    def test_tf_model(self):
        estimator = self.estimator_type(model=self.tf_model,
                                        epochs=3,
                                        nn_batch_size=2,
                                        verbosity=0
                                        )
        estimator.fit(self.X, self.y)

        estimator.save("keras_example_saved_model")

        reload_estinator = self.estimator_type()
        reload_estinator.load("keras_example_saved_model")

        np.testing.assert_array_almost_equal(estimator.predict(self.X), reload_estinator.predict(self.X), decimal=4)

        # remove saved keras files
        for fname in os.listdir("."):
            if fname.startswith("keras_example_saved_model"):
                os.remove(os.path.join(".", fname))


class KerasBaseRegressorTest(KerasBaseClassifierTest):

    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)

        self.model = Sequential()
        self.model.add(Dense(5, input_dim=self.X.shape[1], activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        self.estimator_type = KerasBaseRegressor

        inputs = tf.keras.Input(shape=(self.X.shape[1],))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)(x)
        self.tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.tf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
