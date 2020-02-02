from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseRegressor
import photonai.modelwrapper.keras_base_models as keras_dnn_base_model


class KerasDnnRegressor(KerasDnnBaseModel, KerasBaseRegressor):

    def __init__(self,
                 hidden_layer_sizes: int = [],
                 learning_rate: float = 0.01,
                 loss: str = "mean_squared_error",
                 epochs: int = 10,
                 nn_batch_size: int = 64,
                 metrics: list = ['mean_squared_error'],
                 early_stopping: bool = True,
                 eaSt_patience=20,
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

        super(KerasDnnRegressor, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                                target_activation="linear",
                                                target_dimension=1,
                                                learning_rate=learning_rate,
                                                loss=loss,
                                                metrics=metrics,
                                                early_stopping=early_stopping,
                                                eaSt_patience=eaSt_patience,
                                                batch_normalization=batch_normalization,
                                                verbosity=verbosity,
                                                dropout_rate=dropout_rate,  # list or float
                                                activations=activations,  # list or str
                                                optimizer=optimizer)  # list or keras.optimizer)

    @property
    def target_activation(self):
        return self._target_activation

    @target_activation.setter
    def target_activation(self, value):
        if value == "linear":
            self._target_activation = value
        else:
            raise ValueError("The subclass of KerasBaseRegressor does not allow to use another "
                             "target_activation. Please use 'linear' like default.")

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
        self.encode_targets(y)
        self.create_model(X.shape[1])
        super(KerasDnnBaseModel, self).fit(X, y, reload_weights=True)
