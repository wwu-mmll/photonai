import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class TFDNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, gd_alpha=0.1):
        self.gd_alpha = gd_alpha
        self.sess = tf.InteractiveSession()

        self.w = None
        self.b = None

    def fit(self, X, y, **kwargs):

        y = self.dense_to_one_hot(y, 10)

        # setup placeholders,variable and model
        x = tf.placeholder(tf.float32, [None, X.shape[1]])
        self.w = tf.Variable(tf.zeros([X.shape[1], 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # model
        model_y = tf.nn.softmax(tf.matmul(x, self.w) + self.b)

        # labels
        y_ = tf.placeholder(tf.float32, [None, 10])

        # loss function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(model_y), reduction_indices=[1]))

        # what to do in training
        train_step = tf.train.GradientDescentOptimizer(self.gd_alpha).minimize(cross_entropy)

        # GO!
        tf.global_variables_initializer().run()

        batch_size = 100
        for i in range(np.floor_divide(X.shape[0], batch_size)):
            batch_xs = X[i*batch_size:i*batch_size + batch_size - 1, :]
            batch_ys = y[i*batch_size:i*batch_size + batch_size - 1]
            self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        return self

    def predict(self, X):
        X = tf.cast(X, tf.float32)
        predicted_y = self.sess.run(tf.nn.softmax(tf.matmul(X, self.w) + self.b))
        return np.argmax(predicted_y, 1)

    # def fit_predict(self, X, y, **kwargs):
    #     return self.predict(X)

    # Helper functions
    #######################################################
    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

        # evaluation
        # self.correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
