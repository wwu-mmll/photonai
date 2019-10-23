import numpy as np
from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseClassifier
import photonai.modelwrapper.keras_base_models as keras_dnn_base_model


class KerasDnnClassifier(KerasDnnBaseModel, KerasBaseClassifier):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: int =[],
                 learning_rate: float = 0.1,
                 loss: str = "",
                 epochs: int = 100,
                 batch_size: int =64,
                 metrics: list = ['accuracy'],
                 early_stopping: bool = True,
                 eaSt_patience=20,
                 batch_normalization: bool = True,
                 verbosity=0,
                 dropout_rate=0.2,  # list or float
                 activations='tanh',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self.multi_class = multi_class

        if not loss:
            if self.multi_class:
                loss = "categorical_crossentropy"
            else:
                loss = "binary_crossentropy"

        super(KerasDnnClassifier, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                                target_activation="softmax",
                                                learning_rate=learning_rate,
                                                loss=loss,
                                                metrics=metrics,
                                                early_stopping=early_stopping,
                                                eaSt_patience=eaSt_patience,
                                                batch_normalization=batch_normalization,
                                                dropout_rate=dropout_rate,  # list or float
                                                activations=activations,  # list or str
                                                optimizer=optimizer)  # list or keras.optimizer)


        super(KerasBaseClassifier, self).__init__(model=None, epochs=epochs ,batch_size=batch_size ,verbosity=verbosity)

    @property
    def target_activation(self):
        return self._target_activation

    @target_activation.setter
    def target_activation(self, value):
        if value == "softmax":
            self._target_activation = value
        else:
            raise ValueError("The Classifcation subclass of KerasDnnBaseModel does not allow to use another "
                             "target_activation. Please use 'softmax' like default.")

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        if self.multi_class and value in keras_dnn_base_model.get_loss_allocation()["multi_classification"]:
            self._loss = value
        elif (not self.multi_class) and value in keras_dnn_base_model.get_loss_allocation()["binary_classification"]:
            self._loss = value
        else:
            raise ValueError("Loss function is not supported. Feel free to use upperclass without restrictions.")

    def encode_targets(self, y):
        class_nums = len(np.unique(y))
        if not self.multi_class:
            if class_nums != 2:
                raise ValueError("Can not use binary classification with more or less than two target classes.")
            self.target_dimension = 1
        else:
            self.target_dimension = len(np.unique(y))

        return super(KerasDnnClassifier, self).encode_targets(y)

    def fit(self,X, y):

        self.encode_targets(y)

        self.model = self.create_model(X.shape[1])

        super(KerasDnnBaseModel, self).fit(X,y)

