'''Custom Layers'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from .cactivations import get as cget


class BahdanauAttention(Layer):
    '''
    BahdanauAttention:
    Implemented based on below paper
        https://arxiv.org/pdf/1409.0473.pdf
        attention_weights = probability_fn((Va * tanh(Wa*Ht+Ua*Hs+b))/sqrt(scaling_factor))
    # Arguments
        units = number of hidden units to use.
        probability_fn = probability function to get probabilities(weights for attention)
                         You can use 'softmax' or 'hardmax' or 'sparsemax' or any custom
                         function which takes input distribution and returns probability dist.
        dropout_rate = dropout for attention weights (between 0 and 1, 0 - no dropout).
        return_aweights = Bool, whether to return attention weights or not.
        scaling_factor = int/float to scale the score vector. default None=1
        weights_initializer = initializer for weight matrix
        bias_initializer = initializer for bias values
        weights_constraint = Constraint function applied to the weights
        bias_constraint = Constraint function applied to the bias
    # Returns
        context_vector = context vector after applying attention.
        attention_weights = attention weights only if `return_aweights=True`.

    # Inputs to the layer
        inputs = dictionary with keys "enocderHs", "decoderHt".
                enocderHs = all the encoder hidden states,
                            shape - (Batchsize, encoder_seq_len, enc_hidden_size)
                 decoderHt = hidden state of decoder at that timestep,
                            shape - (Batchsize, dec_hidden_size)
        mask = You can apply mask for padded values or any custom values
               while calculating attention.
               if you are giving mask for encoder and deocoder then you have
               to give a dict similar to inputs. (keys: enocderHs, decoderHt)
               else you can give only for enocoder normally.(one tensor)
               mask shape should be (Batchsize, encoder_seq_len)
    '''
    def __init__(self, units,
                 probability_fn='softmax',
                 dropout_rate=0,
                 return_aweights=False,
                 scaling_factor=None,
                 weights_initializer='he_normal',
                 bias_initializer='zeros',
                 weights_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = ""
        super(BahdanauAttention, self).__init__(**kwargs)

        self.units = units
        self.dropout_rate = dropout_rate
        self.return_aweights = return_aweights
        self.scaling_factor = scaling_factor
        self.probability_fn = cget(probability_fn)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.weights_constraint = weights_constraint
        self.bias_constraint = bias_constraint
        self._wa = layers.Dense(self.units, use_bias=False,\
            kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer,\
                kernel_constraint=self.weights_constraint, bias_constraint=self.bias_constraint,\
                    name=self.name+"Wa")
        self._ua = layers.Dense(self.units,\
            kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer,\
                kernel_constraint=self.weights_constraint, bias_constraint=self.bias_constraint,\
                    name=self.name+"Ua")
        self._va = layers.Dense(1, kernel_initializer=self.weights_initializer,\
            bias_initializer=self.bias_initializer, kernel_constraint=self.weights_constraint,\
                bias_constraint=self.bias_constraint, name=self.name+"Va")
        self.supports_masking = True


    def build(self, input_shape):
        '''build'''
        assert isinstance(input_shape, dict)

        shape_en, shape_dc = input_shape['enocderHs'], input_shape['decoderHt']

        assert len(shape_en) == 3, "Encoder Hiddenstates/output should be 3 dim \
        ( B x T x H ), but got {} dim".format(len(shape_en))

        assert len(shape_dc) == 2, "Decoder Hidden/output should be 2 \
        dim (B x H), but got {} dim".format(len(shape_dc))

        self.built = True # pylint: disable=W0201

    def call(self, inputs, mask=None):
        '''call'''
        assert isinstance(inputs, dict)

        if ('enocderHs' not in inputs.keys())or ('decoderHt' not in inputs.keys()):
            raise ValueError("Input to the layer must be a dict with \
            keys=['enocderHs','decoderHt']")
        if isinstance(mask, dict):
            mask_enc = mask.get('enocderHs', None)
            mask_dec = mask.get('decoderHt', None)
        else:
            mask_enc = mask
            mask_dec = None

        enc_out, dec_prev_hs = tf.cast(inputs['enocderHs'], tf.float32), \
            tf.cast(inputs['decoderHt'], tf.float32)
        if mask_dec is not None:
            dec_prev_hs = dec_prev_hs * tf.cast(mask_dec, dec_prev_hs.dtype)

        # decprev_hs - Decoder hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        dec_hidden_with_time_axis = tf.expand_dims(dec_prev_hs, 1)

        # score shape == (batch_size, max_length)
        score = self._va(tf.nn.tanh(self._wa(dec_hidden_with_time_axis) + self._ua(enc_out)))
        score = tf.squeeze(score, [2])

        if self.scaling_factor is not None:
            score = score/tf.sqrt(self.scaling_factor)

        if mask_enc is not None:
            score = score + (tf.cast(mask_enc, score.dtype)*-1e9)

        # attention_weights shape == (batch_size, max_length)
        attention_weights = self.probability_fn(score, axis=-1)
        #(batch_size, 1, max_length)
        attention_weights = tf.expand_dims(attention_weights, 1)

        if self.dropout_rate != 0:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)

        #context_vector shape (batch_size, hidden_size)
        context_vector = tf.matmul(attention_weights, enc_out)
        context_vector = tf.squeeze(context_vector, 1, name="context_vector")

        if self.return_aweights:
            return context_vector, tf.squeeze(attention_weights, 1, name='attention_weights')
        return context_vector

    def compute_output_shape(self, input_shape):
        '''compute output shape'''
        assert isinstance(input_shape, dict)
        shape_en = input_shape['enocderHs']
        if self.return_aweights:
            output_shape = [(shape_en[0], shape_en[2]), (shape_en[0], shape_en[1])]
            return output_shape
        output_shape = shape_en[0], shape_en[2]
        return output_shape


class LuongeAttention:
    '''
    LuongeAttention
    Implemented based on below paper
    https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self, units,
                 attention_type,
                 probability_fn='softmax',
                 dropout_rate=0,
                 return_aweights=False,
                 scaling_factor=None,
                 weights_initializer='he_normal',
                 bias_initializer='zeros',
                 weights_constraint=None,
                 bias_constraint=None, **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = ""
        super(LuongeAttention, self).__init__(**kwargs)

        self.units = units
        self.attention_type = attention_type
        self.probability_fn = cget(probability_fn)
        self.dropout_rate = dropout_rate
        self.return_aweights = return_aweights
        self.scaling_factor = scaling_factor
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.weights_constraint = weights_constraint
        self.bias_constraint = bias_constraint
        