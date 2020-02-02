import numpy as np
from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseClassifier
import photonai.modelwrapper.keras_base_models as keras_dnn_base_model


class KerasDnnClassifier(KerasDnnBaseModel, KerasBaseClassifier):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: int =[],
                 learning_rate: float = 0.01,
                 loss: str = "",
                 epochs: int = 100,
                 nn_batch_size: int =64,
                 metrics: list = ['accuracy'],
                 early_stopping: bool = False,
                 eaSt_patience=20,
                 batch_normalization: bool = True,
                 verbosity=1,
                 dropout_rate=0.2,  # list or float
                 activations='relu',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.nn_batch_size = nn_batch_size

        super(KerasDnnClassifier, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                   target_activation="sigmoid",
                                   learning_rate=learning_rate,
                                   loss=loss,
                                   metrics=metrics,
                                   early_stopping=early_stopping,
                                   eaSt_patience=eaSt_patience,
                                   batch_normalization=batch_normalization,
                                   dropout_rate=dropout_rate,  # list or float
                                   activations=activations,  # list or str
                                   optimizer=optimizer,
                                   verbosity=verbosity)  # list or keras.optimizer)


    @property
    def multi_class(self):
        return self._multi_class

    @multi_class.setter
    def multi_class(self, value):
        self._multi_class = value

        if not self.loss or self.loss in ["categorical_crossentropy", "binary_crossentropy"]:
            if value:
                self.loss = "categorical_crossentropy"
            else:
                self.loss = "binary_crossentropy"

    @property
    def target_activation(self):
        return self._target_activation

    @target_activation.setter
    def target_activation(self, value):
        if value == "sigmoid":
            self._target_activation = value
        else:
            raise ValueError("The Classifcation subclass of KerasDnnBaseModel does not allow to use another "
                             "target_activation. Please use 'sigmoid' like default.")

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        if value == "":
            if self._multi_class:
                self._loss = "categorical_crossentropy"
            else:
                self._loss = "binary_crossentropy"
        elif self._multi_class and value in keras_dnn_base_model.get_loss_allocation()["multi_classification"]:
            self._loss = value
        elif (not self._multi_class) and value in keras_dnn_base_model.get_loss_allocation()["binary_classification"]:
            self._loss = value
        else:
            raise ValueError("Loss function is not supported. Feel free to use upperclass without restrictions.")

    def encode_targets(self, y):
        class_nums = len(np.unique(y))
        if not self.multi_class:
            if class_nums != 2 and self.loss == 'binary_crossentropy':
                raise ValueError("Can not use binary classification with more or less than two target classes.")
            self.target_dimension = 1
        else:
            self.target_dimension = len(np.unique(y))

        return super(KerasDnnClassifier, self).encode_targets(y)

    def fit(self,X, y):
        y = self.encode_targets(y)
        self.create_model(X.shape[1])
        super(KerasDnnClassifier, self).fit(X,y, reload_weights=True)

