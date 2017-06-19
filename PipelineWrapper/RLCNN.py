import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class RLCNN(BaseEstimator, ClassifierMixin):

    def __init__(self, image_width=24*12, image_height=5*60, num_labels=5, learning_rate=1e-4,
                 number_densely_neurons=1024, patch_size=100, num_features=32, reduction_1=6, reduction_2=2):

        self.image_width = image_width
        self.image_height = image_height
        self.num_labels = num_labels

        self.learning_rate = learning_rate

        self.number_densely_neurons = number_densely_neurons
        self.patch_size = patch_size
        self.num_features = num_features
        self.reduction_1 = reduction_1
        self.reduction_2 = reduction_2

        self.x = None
        self.y_ = None
        self.y_conv = None

    def fit(self, data, targets):

        targets = RLCNN.dense_to_one_hot(targets, self.num_labels)

        self.x = tf.placeholder(tf.float32, [None, self.image_width*self.image_height])
        # 1st dimension: number of data entries
        # 2nd and 3rd dimension: image width and height
        # 4th dimension: number of color channels.
        x_image = tf.reshape(self.x, [-1, self.image_width, self.image_height, 1])

        # labels
        self.y_ = tf.placeholder(tf.float32, [None, self.num_labels])

        # first convolution
        # compute 32 features for each 5x5 patch.
        W_conv1 = RLCNN.weight_variable([self.patch_size, self.patch_size, 1, self.num_features])
        b_conv1 = RLCNN.bias_variable([self.num_features])
        # filter & remove negative numbers
        h_conv1 = tf.nn.relu(RLCNN.conv2d(x_image, W_conv1) + b_conv1)
        # max_pool_2x2 => reduce from 28x28 to 14x14.
        h_pool1 = RLCNN.max_pool_2x2(h_conv1, self.reduction_1)

        # second convolution
        # compute 64 features for each 5x5 patch.
        W_conv2 = RLCNN.weight_variable([self.patch_size, self.patch_size, self.num_features, 2*self.num_features])
        b_conv2 = RLCNN.bias_variable([2*self.num_features])
        h_conv2 = tf.nn.relu(RLCNN.conv2d(h_pool1, W_conv2) + b_conv2)
        # reduce from 14x14 to 7x7
        h_pool2 = RLCNN.max_pool_2x2(h_conv2, self.reduction_2)

        # densely connected layer
        # weights = 7x7 image and 64 features, 1024 neurons
        final_size = int((self.image_width/self.reduction_1/self.reduction_2) *
                         (self.image_height/self.reduction_1/self.reduction_2) * 2 * self.num_features)
        W_fc1 = RLCNN.weight_variable([final_size, self.number_densely_neurons])
        b_fc1 = RLCNN.bias_variable([self.number_densely_neurons])

        h_pool2_flat = tf.reshape(h_pool2, [-1, final_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # reduce overfitting
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # final decision layer
        W_fc2 = RLCNN.weight_variable([self.number_densely_neurons, self.num_labels])
        b_fc2 = RLCNN.bias_variable([self.num_labels])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        # iteration step
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # success rate
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # GO!
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            train_accuracy = accuracy.eval(feed_dict={self.x: data, self.y_: targets, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # train
            train_step.run(feed_dict={self.x: data, self.y_: targets, keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            self.x: data, self.y_: targets, keep_prob: 1.0}))

        return self

    # def transform(self, data):
    #     return data

    def predict(self, data):
        classification = tf.run(self.y_conv, feed_dict={
            self.x: data})
        return classification

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x, reduction_size):
        return tf.nn.max_pool(x, ksize=[1, reduction_size, reduction_size, 1],
                            strides=[1, reduction_size, reduction_size, 1], padding='SAME')









