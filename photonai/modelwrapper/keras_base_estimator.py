from keras.models import model_from_json
from sklearn.base import BaseEstimator
from photonai.photonlogger.logger import logger
from sklearn.model_selection import ShuffleSplit

class KerasBaseEstimator(BaseEstimator):
    """
    base class for all Keras wrappers
    """

    def __init__(self,
                 model=None,
                 epochs: int = 10,
                 batch_size: int = 64,
                 verbosity: int = 0):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbosity = verbosity

    def fit(self, X, y):
        # prepare target values
        # Todo: early stopping

        y = self.encode_targets(y)

        # use callbacks only when size of training set is above 100
        if X.shape[0] > 100:
            # get pseudo validation set for keras callbacks
            splitter = ShuffleSplit(n_splits=1, test_size=0.2)
            for train_index, val_index in splitter.split(X):
                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]

            # fit the model
            results = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=self.verbosity)
        else:
            # fit the model
            logger.warn('Cannot use Keras Callbacks because of small sample size.')
            results = self.model.fit(X, y, batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=self.verbosity)

        return self

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        return self.model.predict(X, batch_size=self.batch_size)

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
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(filename + ".h5")
        self.model = loaded_model

    def load_nounzip(self, archive, element_info):
        # load json and create model
        loaded_model_json = archive.read(element_info['filename'] + '.json') #.decode("utf-8")
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        # ToDo: fix loading hdf5 without unzipping first
        loaded_weights = archive.read(element_info['filename'] + '.h5')
        loaded_model.load_weights(loaded_weights)

        self.model = loaded_model
