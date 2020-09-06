import photonai.modelwrapper.keras_base_models as keras_dnn_base_model

from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseRegressor
from photonai.photonlogger import logger


class KerasDnnRegressor(KerasDnnBaseModel, KerasBaseRegressor):

    def __init__(self,
                 hidden_layer_sizes: int = None,
                 learning_rate: float = 0.01,
                 loss: str = "mean_squared_error",
                 epochs: int = 10,
                 nn_batch_size: int = 64,
                 metrics: list = None,
                 validation_split: float = 0.1,
                 callbacks: list = None,
                 batch_normalization: bool = True,
                 verbosity=0,
                 dropout_rate=0.2,  # list or float
                 activations='relu',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.epochs =epochs
        self.nn_batch_size = nn_batch_size
        self.validation_split = validation_split

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['mean_squared_error']

        super(KerasDnnRegressor, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                                target_activation="linear",
                                                target_dimension=1,
                                                learning_rate=learning_rate,
                                                loss=loss,
                                                metrics=metrics,
                                                batch_normalization=batch_normalization,
                                                verbosity=verbosity,
                                                dropout_rate=dropout_rate,  # list or float
                                                activations=activations,  # list or str
                                                optimizer=optimizer)  # list or keras.optimizer)

    @property
    def target_activation(self):
        return "linear"

    @target_activation.setter
    def target_activation(self, value):
        if value != "linear":
            msg = "The subclass of KerasBaseRegressor does not allow to use another " \
                  "target_activation. Please use 'linear' like default."
            logger.error(msg)
            raise ValueError(msg)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        if value in keras_dnn_base_model.get_loss_allocation()["regression"]:
            self._loss = value
        else:
            raise ValueError("Loss function is not supported. Feel free to use upperclass without restrictions.")

    def fit(self, X, y):
        self.create_model(X.shape[1])
        super(KerasDnnBaseModel, self).fit(X, y, reload_weights=True)
