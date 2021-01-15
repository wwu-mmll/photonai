import warnings
import numpy as np
import keras
from typing import Union
from keras.utils import to_categorical
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Optimizer, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam, SGD
from keras.activations import softmax, softplus, selu, sigmoid, softsign, hard_sigmoid, elu, relu, tanh, \
    linear, exponential
from sklearn.base import ClassifierMixin, RegressorMixin

from photonai.photonlogger.logger import logger
from photonai.modelwrapper.keras_base_estimator import KerasBaseEstimator

__supported_optimizers__ = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'adam': Adam,
    'adamax': Adamax,
    'nadam': Nadam
}
__supported_activations__ = {
    'softmax': softmax,
    'softplus': softplus,
    'selu': selu,
    'sigmoid': sigmoid,
    'softsign': softsign,
    'hard_sigmoid': hard_sigmoid,
    'elu': elu,
    'relu': relu,
    'tanh': tanh,
    'linear': linear,
    'exponential': exponential
}

__allocation_loss_functions__ = {
    'regression': ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error'],
    'binary_classification': ['binary_crossentropy', 'hinge', 'squared_hinge'],
    'multi_classification': ['categorical_crossentropy', 'sparse_categorical_crossentropy',
                             'kullback_leibler_divergence']
}


class KerasBaseClassifier(KerasBaseEstimator, ClassifierMixin):

    def __init__(self,
                 model=None,
                 epochs: int = 10,
                 callbacks: list = None,
                 validation_split: float = 0.1,
                 nn_batch_size: int = 33,
                 multi_class: bool = True,
                 verbosity: int = 0):
        super(KerasBaseClassifier, self).__init__(model=model,
                                                  epochs=epochs,
                                                  callbacks=callbacks,
                                                  validation_split=validation_split,
                                                  nn_batch_size=nn_batch_size,
                                                  verbosity=verbosity)
        self.multi_class = multi_class

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=self.nn_batch_size)
        if self.multi_class:
            max_index = np.argmax(predict_result, axis=1)
        else:
            max_index = np.array([val > 0.5 for val in predict_result])
        return max_index

    def encode_targets(self, y):
        if self.multi_class:
            return to_categorical(y)
        else:
            return y


class KerasBaseRegressor(KerasBaseEstimator, RegressorMixin):

    def __init__(self,
                 model=None,
                 epochs: int = 10,
                 validation_split: float = 0.1,
                 nn_batch_size: int = 64,
                 callbacks: list = None,
                 verbosity: int = 0):
        super(KerasBaseRegressor, self).__init__(model=model,
                                                 epochs=epochs,
                                                 validation_split=validation_split,
                                                 nn_batch_size=nn_batch_size,
                                                 callbacks=callbacks,
                                                 verbosity=verbosity)

    def predict(self, X):
        return np.array([val[0] for val in self.predict_proba(X)])


class KerasDnnBaseModel(KerasBaseEstimator):

    def __init__(self, hidden_layer_sizes: list = None,
                 target_dimension: int = 2,
                 target_activation: str = "softmax",
                 learning_rate: float = 0.1,
                 loss: str = "categorical_crossentropy",
                 metrics: list = None,
                 batch_normalization: bool = True,
                 verbosity: int = 0,
                 dropout_rate: Union[list, float] = 0.2,
                 activations: Union[list, str] = 'relu',
                 optimizer="adam"  # list or keras.optimizer
                 ):

        self._hidden_layer_sizes = None
        self._dropout_rate = None
        self._target_activation = None
        self._optimizer = None
        self._loss = None
        self._activations = None
        self._metrics = None

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.target_dimension = target_dimension
        self.target_activation = target_activation
        self.activations = activations
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.batch_normalization = batch_normalization
        self.metrics = metrics

        self.verbosity = verbosity

        self.model = None

    @property
    def hidden_layer_sizes(self):
        """
        Getter for attribute dropout.
        :return:
        """
        return self._hidden_layer_sizes

    @hidden_layer_sizes.setter
    def hidden_layer_sizes(self, value):
        """
        Setter for dropout_rate. Checks if strategy is supported.
        :param value:
        :return:
        """
        self._hidden_layer_sizes = value
        if not value:
            return
        layer_count = len(value)
        if isinstance(self.activations, list):
            if len(self.activations) != layer_count:
                raise ValueError("activations length missmatched layer length.")
        else:
            self._activations = [self.activations] * layer_count
        if isinstance(self.dropout_rate, list):
            if len(self.dropout_rate) != layer_count:
                raise ValueError("dropout_rate length missmatched layer length.")
        else:
            self._dropout_rate = [self.dropout_rate] * layer_count

    @property
    def dropout_rate(self):
        """
        Getter for attribute dropout.
        :return:
        """
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        """
        Setter for dropout_rate. Checks if strategy is supported.
        :param value:
        :return:
        """
        if not type(value) in [list, float]:
            raise ValueError("Dropout type is not supported. Please use one of [list, float]")
        else:
            if not self._hidden_layer_sizes:
                self._dropout_rate = value
            else:
                if type(value) == float:
                    self._dropout_rate = [value]*len(self.hidden_layer_sizes)
                    msg = "Dropout with type float converted to type list."
                    logger.warning(msg)
                    warnings.warn(msg)
                elif len(value) != len(self.hidden_layer_sizes):
                    raise ValueError("Dropout length missmatched layer length.")
                else:
                    self._dropout_rate = value

    @property
    def activations(self):
        """
        Getter for attribute act_func.
        :return:
        """
        return self._activations

    @activations.setter
    def activations(self, value):
        """
        Setter for act_func. Checks if strategy is supported.
        :param value:
        :return:
        """
        if not type(value) in [list, str]:
            raise ValueError("act_func type is not supported. Please use one of [list, float]")
        else:
            if not self._hidden_layer_sizes:
                self._activations = value
            else:
                if type(value) == str:
                    if value in __supported_activations__.keys():
                        self._activations = [value]*len(self.hidden_layer_sizes)
                        msg = "activations with type str converted to type list."
                        logger.warning(msg)
                        warnings.warn(msg)
                    else:
                        raise ValueError(
                            "activations not supported. Please use one of: " + str(__supported_activations__.keys()))
                elif len(value) != len(self.hidden_layer_sizes):
                    raise ValueError("activations length missmatched layer length.")
                elif any(act not in __supported_activations__.keys() for act in value):
                    raise ValueError("activations not supported. Please use one of: "+str(__supported_activations__.keys()))
                else:
                    self._activations = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if isinstance(value, Optimizer):
            self._optimizer = value
        if value.lower() not in __supported_optimizers__.keys():
            raise ValueError("Optimizer is not supported by keras. Please use one of: "+str(__supported_optimizers__))
        else:
            self._optimizer = __supported_optimizers__[value.lower()](lr=self.learning_rate)

    @property
    def target_activation(self):
        return self._target_activation

    @target_activation.setter
    def target_activation(self, value):
        if value in __supported_activations__.keys():
            self._target_activation = value
        else:
            raise ValueError("target_activation not supported. Please use one of: " + str(__supported_activations__.keys()))

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        try:
            keras.losses.get(value)
            self._loss = value
        except ValueError:
            raise ValueError("Unknown loss function:" + value)

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        # TODO: check metrics values!
#        try:
#            all(keras.metrics.get(metric) for metric in value)
            self._metrics = value
#        except ValueError:
#            raise ValueError("Unknown metric function in:" + str(value))

    def create_model(self, input_size):

        self.model = Sequential()
        for i, size in enumerate(self.hidden_layer_sizes):
            if i == 0:
                self.model.add(Dense(size, input_dim=input_size, activation=self.activations[i]))
            else:
                self.model.add(Dense(size, activation=self.activations[i]))
            self.model.add(Dropout(rate=self.dropout_rate[i]))

            if self.batch_normalization == 1:
                self.model.add(BatchNormalization())

        self.model.add(Dense(self.target_dimension, activation=self.target_activation))

        # Compile model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.init_weights = self.model.get_weights()

        return self


def get_loss_allocation():
    return __allocation_loss_functions__
