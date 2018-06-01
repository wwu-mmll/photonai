from keras.models import model_from_json

class KerasBaseEstimator(object):
    """base class for all Keras wrappers
    """
    def __init__(self):
        self.model = None

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
        loaded_model.load_weights(filename + ".h5")
        # load weights into new model
        self.model = loaded_model
