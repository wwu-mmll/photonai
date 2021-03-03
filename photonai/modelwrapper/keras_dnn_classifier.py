import numpy as np
from typing import Union
from keras.optimizers import Optimizer

from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseClassifier
import photonai.modelwrapper.keras_base_models as keras_dnn_base_model


class KerasDnnClassifier(KerasDnnBaseModel, KerasBaseClassifier):
    """Wrapper class for a classification-based Keras model.

    See [Keras API](https://keras.io/api/).

    Example:
        ``` python
        PipelineElement('KerasDnnClassifier',
                        hyperparameters={'hidden_layer_sizes': Categorical([[10, 8, 4], [20, 15, 5]]),
                                         'dropout_rate': Categorical([0.5, [0.5, 0.2, 0.1]])},
                        activations='relu',
                        nn_batch_size=32,
                        multi_class=True,
                        verbosity=1)
        ```

    """
    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 learning_rate: float = 0.01,
                 loss: str = "",
                 epochs: int = 100,
                 nn_batch_size: int = 64,
                 metrics: list = None,
                 callbacks: list = None,
                 validation_split: float = 0.1,
                 verbosity: int = 1,
                 dropout_rate: Union[float, list] = 0.2,
                 activations: Union[str, list] = 'relu',
                 optimizer: Union[Optimizer, str] = "adam"):
        """
        Initialize the object.

        Parameters:
            multi_class:
                Enables multi_target learning.

            hidden_layer_sizes:
                Number of perceptrons per layer.

            learning_rate:
                Step size of the learning adjustment.

            loss:
                Loss function.

            epochs:
                Number of arbitrary cutoffs, generally defined as
                "one pass over the entire dataset", used to separate training into distinct phases,
                which is useful for logging and periodic evaluation.

            nn_batch_size:
                Typically the batch_size. A batch is a set of nn_batch_size samples.
                The samples in a batch are processed independently, in parallel.
                If training, a batch results in only one update to the model.

            metrics:
                List of evaluate metrics.

            callbacks:
                Within Keras, there is the ability to add callbacks specifically designed
                to be run at the end of an epoch. Examples of these
                are learning rate changes and model checkpointing (saving).

            validation_split:
                Split size of validation set.

            verbosity:
                The level of verbosity, 0 is least talkative and
                gives only warn and error, 1 gives adds info and 2 adds debug.

            dropout_rate:
                A Dropout layer applies random dropout and rescales the output.
                In inference mode, the same layer does nothing.
                Float -> added behind each layer
                List -> Same size as hidden_layer_size

            activations:
                Activation function.

            optimizer:
                Optimization algorithm.

        """
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs = epochs
        self.nn_batch_size = nn_batch_size
        self.validation_split = validation_split

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']

        super(KerasDnnClassifier, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                                 target_activation="softmax",
                                                 learning_rate=learning_rate,
                                                 loss=loss,
                                                 metrics=metrics,
                                                 dropout_rate=dropout_rate,
                                                 activations=activations,
                                                 optimizer=optimizer,
                                                 verbosity=verbosity)

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

    def _calc_target_dimension(self, y):
        class_nums = len(np.unique(y))
        if not self.multi_class:
            if class_nums != 2 and self.loss == 'binary_crossentropy':
                raise ValueError("Can not use binary classification with more or less than two target classes.")
            self.target_dimension = 1
        else:
            self.target_dimension = len(np.unique(y))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Starting the learning process of the neural network.

        Parameters:
            X:
                The input samples with shape [n_samples, n_features].

            y:
                The input targets with shape [n_samples, 1].

        """
        self._calc_target_dimension(y)
        self.create_model(X.shape[1])
        super(KerasDnnClassifier, self).fit(X, y)
        return self
