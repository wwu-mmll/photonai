from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from numpy import *
from matplotlib.pyplot import *
from scipy import linalg

class PyESNWrapper():

    def __init__(self, reservoir_size=1000, leaking_rate=0.3, regularization_coeff=1e-8, spectral_radius=1.25,
                 in_size=1, out_size=1, init_len=100):

        self.in_size = in_size
        self.out_size = out_size
        self.res_size = reservoir_size
        self.init_len = init_len
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.regularization_coeff = regularization_coeff
        self.activation_function = tanh
        self.w_in = None
        self.w_out = None
        self.w = None
        self.x = None

        random.seed(42)

    def fit(self, data, targets):

        if self.spectral_radius >= 1 and np.count_nonzero(data == 0) > 0:
            raise ValueError('ESN wont work: with a tanh sigmoid, the ESP is violated for zero input if the spectral radius of the reservoir weight matrix is larger than unity')

        self.in_size = data.shape[1]
        train_len = data.shape[0]
        self.w_in = (random.rand(self.res_size, 1 + self.in_size) - 0.5) * 1
        self.w = random.rand(self.res_size, self.res_size) - 0.5

        # Option 1 - direct scaling (quick&dirty, reservoir-specific):
        # W *= 0.135
        # Option 2 - normalizing and setting spectral radius (correct, slow):
        print('Computing spectral radius...', )
        rhoW = max(abs(linalg.eig(self.w)[0]))
        print('..done.')
        self.w *= self.spectral_radius / rhoW

        # allocated memory for the design (collected states) matrix
        X = zeros((1 + self.in_size + self.res_size, train_len - self.init_len))
        # set the corresponding target matrix directly
        Yt = targets.T

        # run the reservoir with the data and collect X
        self.x = zeros((self.res_size, 1))
        for t in range(train_len):
            u = data[t, :]
            # stacked_vectors = vstack((np.ones((u.shape[0],1)).flatten(), u))
            bias_u = self.add_bias_unit(u)
            self.calculate_state(bias_u)
            if t >= self.init_len:
                stack_input_and_output = concatenate((bias_u, self.x))
                X[:, t - self.init_len] = stack_input_and_output

        # train the output
        X_T = X.T
        y_times_transposed_states = dot(Yt, X_T)
        state_times_transposed_states = dot(X, X_T)
        regularization_matrix = self.regularization_coeff * eye(1 + self.in_size + self.res_size)
        # nr_of_infs_reg = np.count_nonzero(~np.isnan(data))
        # nr_infs_reg = np.count_nonzero(np.isinf(regularization_matrix))
        nr_infs_state = np.count_nonzero(np.isinf(state_times_transposed_states))
        inverse = linalg.inv(state_times_transposed_states + regularization_matrix)
        self.w_out = dot(y_times_transposed_states, inverse)
        # Wout = dot( Yt, linalg.pinv(X) )
        return self

    def predict(self, data):
        testLen = data.shape[0]
        # run the trained ESN in a generative mode. no need to initialize here,
        # because x is initialized with training data and we continue from there.
        Y = zeros((testLen, self.out_size))
        for t in range(testLen):
            u = data[t]
            bias_u = self.add_bias_unit(u)
            self.calculate_state(bias_u)
            concated_data = concatenate((bias_u, self.x))
            y = dot(self.w_out, concated_data)
            Y[t, :] = y

            # generative mode:
            # u = y
            # predictive mode
            # u = data[trainLen + t + 1]
        return Y

    def add_bias_unit(self, u):
        return np.insert(u, 0, 1)

    def calculate_state(self, bias_u):
        # x = (1 - a) * x + a * tanh(dot(Win, data_stack) + dot(W, x))
        last_echo = ((1 - self.leaking_rate) * self.x).flatten()
        input_weights_times_data = dot(self.w_in, bias_u).flatten()
        state_times_weights = dot(self.w, self.x).flatten()
        self.x = last_echo + self.leaking_rate * self.activation_function(input_weights_times_data + state_times_weights)


class PyESNRegressor(BaseEstimator, RegressorMixin, PyESNWrapper):
    pass


class PyESNClassifier(BaseEstimator, ClassifierMixin, PyESNWrapper):
    pass
