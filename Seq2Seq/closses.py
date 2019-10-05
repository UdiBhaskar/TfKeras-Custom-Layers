'''loss functions'''
import tensorflow as tf

def seq_crossentropy(y_true, y_pred, from_logits=True, mask_value=0):
    '''sequential cross entropy with masking''''
    y_true = tf.reshape(y_true, shape=(-1, 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits,\
        reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)
