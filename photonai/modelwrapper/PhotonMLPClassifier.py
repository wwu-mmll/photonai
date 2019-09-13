from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin


class PhotonMLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, layer_1: int = 10, layer_2: int = None, layer_3: int = None, layer_4: int = None,
                 activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001,
                 verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-08, n_iter_no_change=10):
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.layer_3 = layer_3
        self.layer_4 = layer_4

        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

        self.model = None

    def fit(self, X, y=None, **kwargs):
        hidden_layer_sizes = list()
        for i, layer in enumerate([self.layer_1, self.layer_2, self.layer_3, self.layer_4]):
            if layer:
                hidden_layer_sizes.append(layer)
            else:
                if i == 0:
                    raise ValueError("Number of neurons in first hidden layer is zero.")
                else:
                    break

        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   activation=self.activation,
                                   solver=self.solver,
                                   alpha=self.alpha,
                                   batch_size=self.batch_size,
                                   learning_rate=self.learning_rate,
                                   learning_rate_init=self.learning_rate_init,
                                   power_t=self.power_t,
                                   max_iter=self.max_iter,
                                   shuffle=self.shuffle,
                                   random_state=self.random_state,
                                   tol=self.tol,
                                   verbose=self.verbose,
                                   warm_start=self.warm_start,
                                   momentum=self.momentum,
                                   nesterovs_momentum=self.nesterovs_momentum,
                                   early_stopping=self.early_stopping,
                                   validation_fraction=self.validation_fraction,
                                   beta_1=self.beta_1,
                                   beta_2=self.beta_2,
                                   epsilon=self.epsilon,
                                   n_iter_no_change=self.n_iter_no_change)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
