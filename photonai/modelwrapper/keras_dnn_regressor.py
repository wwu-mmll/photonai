from typing import Union
import numpy as np
from keras.optimizers import Optimizer
import photonai.modelwrapper.keras_base_models as keras_dnn_base_model

from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseRegressor
from photonai.photonlogger import logger


class KerasDnnRegressor(KerasDnnBaseModel, KerasBaseRegressor):
    """Wrapper class for a regression-based Keras model.

    See [Keras API](https://keras.io/api/).

    Example:
        ``` python
        PipelineElement('KerasDnnRegressor',
                        hyperparameters={'hidden_layer_sizes': Categorical([[18, 14], [30, 5]]),
                                         'dropout_rate': Categorical([0.01, 0.2])},
                        activations='relu',
                        epochs=50,
                        nn_batch_size=64,
                        verbosity=1)
        ```

    """
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
                 verbosity: int = 0,
                 dropout_rate: Union[float, list] = 0.2,
                 activations: Union[str, list] = 'relu',
                 optimizer: Union[Optimizer, str] = "adam"):
        """
        Initialize the object.

        Parameters:
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

            batch_normalization:
                Batch normalization applies a transformation that maintains
                the mean output close to 0 and the output standard deviation close to 1.

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
        self.epochs = epochs
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
                                                dropout_rate=dropout_rate,
                                                activations=activations,
                                                optimizer=optimizer)

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Starting the learning.

        Parameters:
            X:
                The input samples with shape [n_samples, n_features].

            y:
                The input targets with shape [n_samples, 1].

        """
        self.create_model(X.shape[1])
        super(KerasDnnBaseModel, self).fit(X, y)
        return self
