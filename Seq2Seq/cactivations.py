'''Custom activations'''
import warnings
import six
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import deserialize_keras_object

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

def sparsemax(inputs):
    '''sparsemax
    Implementation is based on below paper
    https://arxiv.org/pdf/1602.02068.pdf
    # Arguments
        inputs: Input Tensor
    # Returns
        Tensor, output of the sparsemax transformation
    '''

    inputs = tf.convert_to_tensor(inputs, name='logits')
    shape_inputs = tf.shape(inputs)
    input_sorted, _ = tf.math.top_k(inputs, k=shape_inputs[1])
    input_cumsum = tf.cumsum(input_sorted, axis=-1)
    k = tf.range(1, tf.cast(shape_inputs[1], inputs.dtype)+1, dtype=inputs.dtype)
    input_check = 1 + k*input_sorted > input_cumsum
    k_input = tf.reduce_sum(tf.cast(input_check, tf.int32), axis=1)

    #calculating tau(z)
    k_input_safe = tf.maximum(k_input, 1)
    indices = tf.stack([tf.range(0, shape_inputs[0]), k_input_safe-1], axis=1)
    tau_sum = tf.gather_nd(input_cumsum, indices)
    tau_input = (tau_sum - 1) / tf.cast(k_input, inputs.dtype)

    prob_sparse = tf.maximum(tf.cast(0, inputs.dtype), inputs - tau_input[:, tf.newaxis])

    prob_sparse = tf.where(tf.logical_or(tf.equal(k_input, 0), \
        tf.math.is_nan(input_cumsum[:, -1])), tf.fill([shape_inputs[0], shape_inputs[1]], \
            tf.cast(float("nan"), inputs.dtype)), prob_sparse)
    return prob_sparse

def serialize(activation):
    '''Get the name of activation'''
    return activation.__name__

def deserialize(name, custom_objects=None):
    '''deserialize Keras Object'''
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='activation function')

def get(identifier):
    """Get the `identifier` activation function.
    # Arguments
        identifier: None or str, name of the function.
    # Returns
        The activation function, `linear` if `identifier` is None.
    # Raises
        ValueError if unknown identifier
    """
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)
