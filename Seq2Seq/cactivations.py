'''activations'''
import warnings
import six
import tensorflow as tf
from tensorflow.keras.layers import Layer

def softmax(inputs, axis=-1):
    '''softmax activation function
    # Arguments
        inputs = Input Tensor.
        axis = Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of the softmax transformation
    '''
    return tf.nn.softmax(inputs, axis=axis)

def hardmax(inputs, axis=-1):
    '''hardmax activation function
    # Arguments
        inputs = Input Tensor.
        axis = Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of the hardmax transformation
    '''
    out_shape = tf.shape(inputs)[axis]
    out = tf.one_hot(tf.math.argmax(inputs, axis), out_shape)
    return out

def sparsemax(parameter_list):
    pass
