'''Custom Layers'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


class BahdanauAttention(Layer):
    '''BahdanauAttention'''
    def __init__(self, units, probability_fn=None, **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = ""
        super(BahdanauAttention, self).__init__(**kwargs)

        self.units = units
        self._wa = layers.Dense(self.units, name=self.name+"Wa")
        self._ua = layers.Dense(self.units, name=self.name+"Ua")
        self._va = layers.Dense(1, name=self.name+"Va")
        self.probability_fn = probability_fn

        if probability_fn is None:
            self.probability_fn = tf.nn.softmax
        else:
            self.probability_fn = probability_fn


    def build(self, input_shape):
        '''build'''
        assert isinstance(input_shape, list)

        shape_en, shape_dc = input_shape

        assert len(shape_en) == 3, "Encoder Hiddenstates/output should be 3 dim \
        ( B x T x H ), but got {} dim".format(len(shape_en))

        assert len(shape_dc) == 2, "Decoder Hidden/output should be 2 \
        dim (B x H), but got {} dim".format(len(shape_dc))

        self.built = True # pylint: disable=W0201

    def call(self, inputs, mask=None):
        '''call'''
        assert len(inputs) == 2, "inputs length must be 2 but got {}".format(len(inputs))

        enc_out, dec_prev_hs = inputs
        # decprev_hs - Decoder hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(dec_prev_hs, 1)

        if mask:
            enc_out = enc_out * mask

        # score shape == (batch_size, max_length)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self._va(tf.nn.tanh(self._wa(hidden_with_time_axis) + self._ua(enc_out)))
        score = tf.squeeze(score, [2])
        # attention_weights shape == (batch_size, max_length)
        attention_weights = self.probability_fn(score, axis=-1)
        #(batch_size, max_length, 1)
        attention_weights = tf.expand_dims(attention_weights, 2)
        # context_vector shape (batch_size, 1, hidden_size)
        context_vector = tf.tensordot(attention_weights, enc_out, axes=2)

        context_vector = tf.expand_dims(context_vector, 1)

        return context_vector

    def compute_output_shape(self, input_shape):
        '''compute output shape'''
        assert isinstance(input_shape, list)
        shape_en, _ = input_shape
        output_shape = shape_en[0], 1, shape_en[2]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        '''compute mask'''
        # pylint: disable=W0612,W0613
        return None
        