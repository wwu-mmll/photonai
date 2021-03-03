import warnings
import keras
from sklearn.base import BaseEstimator

from photonai.photonlogger.logger import logger


class KerasBaseEstimator(BaseEstimator):
    """
    base class for all Keras wrappers
    """

    def __init__(self,
                 model: keras.Model = None,
                 epochs: int = 10,
                 nn_batch_size: int = 32,
                 callbacks: list = None,
                 validation_split: float = 0.1,
                 verbosity: int = 0):
        self.model = model
        self.epochs = epochs
        self.nn_batch_size = nn_batch_size
        self.verbosity = verbosity
        self.validation_split = validation_split
        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        # in reloading context self.model could be None
        if self.model is not None:
            # saving initial weights in order to be able to reset the weights before each fit
            # with this approach pre-trained weights can be set
            self.init_weights = model.get_weights()

            # if a non keras layer is in model, it needs to be registered with keras
            for layer in self.model.layers:
                if not layer.__module__.startswith('keras'):
                    keras.utils.get_custom_objects()[type(layer).__name__] = layer.__class__

    def fit(self, X, y):
        # set weights to initial weights to achieve a weight reset
        self.model.set_weights(self.init_weights)

        # allow labels to be encoded before being passed to the model
        # by default self.encode_targets returns identity of y
        y = self.encode_targets(y)

        # use validation split only when size of training set is above 100
        if X.shape[0] < 100 and self.validation_split is not None:
            msg = 'Cannot use validation split because of small sample size.'
            logger.warning(msg)
            warnings.warn(msg)
        self.model.fit(X, y,
                       batch_size=self.nn_batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split if X.shape[0] > 100 else None,
                       callbacks=self.callbacks,
                       verbose=self.verbosity)
        return self

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :return: predicted values, array
        """
        return self.model.predict(X, batch_size=self.nn_batch_size)

    def encode_targets(self, y):
        return y

    def save(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(filename + ".h5")

    def load(self, filename):
        # load json and create model
        json_file = open(filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(filename + ".h5")
        self.model = loaded_model
        self.init_weights = self.model.get_weights()
