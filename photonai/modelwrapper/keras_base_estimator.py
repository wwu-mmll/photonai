from sklearn.base import BaseEstimator
from photonai.photonlogger.logger import logger
import keras
import tensorflow


class KerasBaseEstimator(BaseEstimator):
    """
    base class for all Keras wrappers
    """

    def __init__(self,
                 model=None,
                 epochs: int = 10,
                 nn_batch_size: int = 32,
                 callbacks: list = None,
                 verbosity: int = 0):
        self.model = model
        self.epochs = epochs
        self.nn_batch_size = nn_batch_size
        self.verbosity = verbosity
        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        # saving initial weights in order to be able to reset the weights before each fit
        # with this approach pre-trained weights can be set
        self.init_weights = model.get_weights()

        # if a non keras layer is in model, it needs to be registered with keras
        if isinstance(self.model, keras.Model):
            for layer in self.model.layers:
                if not layer.__module__.startswith('keras'):
                    keras.utils.get_custom_objects()[type(layer).__name__] = layer.__class__
        elif isinstance(self.model, tensorflow.keras.Model):
            for layer in self.model.layers:
                if not layer.__module__.startswith('tensorflow'):
                    tensorflow.keras.utils.get_custom_objects()[type(layer).__name__] = layer.__class__

    def fit(self, X, y, reload_weights: bool = False):
        # set weights to initial weights to achieve a weight reset
        self.model.set_weights(self.init_weights)

        # allow labels to be encoded before being passed to the model
        # by default self.encode_targets returns identity of y
        y = self.encode_targets(y)

        # use callbacks only when size of training set is above 100
        if X.shape[0] > 100:
            # get pseudo validation set for keras callbacks
            # fit the model
            self.model.fit(X, y,
                           batch_size=self.nn_batch_size,
                           validation_split=0.1,
                           epochs=self.epochs,
                           callbacks=self.callbacks,
                           verbose=self.verbosity)
        else:
            # fit the model
            logger.warning('Cannot use Keras Callbacks because of small sample size.')
            self.model.fit(X, y,
                           batch_size=self.nn_batch_size,
                           epochs=self.epochs,
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

    def load_nounzip(self, archive, element_info):
        # load json and create model
        loaded_model_json = archive.read(element_info['filename'] + '.json')  # .decode("utf-8")
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        # ToDo: fix loading hdf5 without unzipping first
        loaded_weights = archive.read(element_info['filename'] + '.h5')
        loaded_model.load_weights(loaded_weights)

        self.model = loaded_model
