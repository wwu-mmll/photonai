"""
Define custom metrics here
"""

import keras.backend as K


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(K.variable(y_true), axis=-1),
                          K.argmax(K.variable(y_pred), axis=-1)),
                  K.floatx())