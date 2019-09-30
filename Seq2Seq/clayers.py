'''Custom Layers'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from .cactivations import get as cget
from .cactivations import serialize

def _attention_score(dec_ht,
                     enc_hs,
                     attention_type,
                     weightwa=None,
                     weightua=None,
                     weightva=None):
    if attention_type == 'bahdanau':
        score = weightva(tf.nn.tanh(weightwa(dec_ht) + weightua(enc_hs)))
        score = tf.squeeze(score, [2])
    elif attention_type == 'dot':
        score = tf.matmul(dec_ht, enc_hs, transpose_b=True)
        score = tf.squeeze(score, 1)
    elif attention_type == 'general':
        score = weightwa(enc_hs)
        score = tf.matmul(dec_ht, score, transpose_b=True)
        score = tf.squeeze(score, 1)    
    elif attention_type == 'concat':
        dec_ht = tf.tile(dec_ht, [1, enc_hs.shape[1], 1])
        score = weightva(tf.nn.tanh(weightwa(tf.concat((dec_ht, enc_hs), axis=-1))))
        score = tf.squeeze(score, 2)
    return score


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
        weights_regularizer = Regularize the weights (U, W, V)
        bias_regularizer = Regularize the bias (b)
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
                 weights_regularizer=None,
                 bias_regularizer=None,
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
        self.weights_initializer = initializers.get(weights_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self._wa = layers.Dense(self.units, use_bias=False,\
            kernel_initializer=weights_initializer, bias_initializer=bias_initializer,\
                kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                    kernel_constraint=weights_constraint, bias_constraint=bias_constraint,\
                        name=self.name+"Wa")
        self._ua = layers.Dense(self.units,\
            kernel_initializer=weights_initializer, bias_initializer=bias_initializer,\
                kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                    kernel_constraint=weights_constraint, bias_constraint=bias_constraint,\
                        name=self.name+"Ua")
        self._va = layers.Dense(1, use_bias=False, kernel_initializer=weights_initializer,\
            kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                bias_initializer=bias_initializer, kernel_constraint=weights_constraint,\
                    bias_constraint=bias_constraint, name=self.name+"Va")
        self.supports_masking = True


    def build(self, input_shape):
        '''build'''
        assert isinstance(input_shape, dict)

        shape_en, shape_dc = input_shape['enocderHs'], input_shape['decoderHt']

        assert len(shape_en) >= 3, "Encoder Hiddenstates/output should be 3 dim or more \
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
        if mask_enc is not None:
            enc_out = enc_out * tf.cast(tf.expand_dims(mask_enc, 2), enc_out.dtype)

        # decprev_hs - Decoder hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        dec_hidden_with_time_axis = tf.expand_dims(dec_prev_hs, 1)

        # score shape == (batch_size, max_length)
        score = _attention_score(dec_ht=dec_hidden_with_time_axis, enc_hs=enc_out,\
                    attention_type='bahdanau', weightwa=self._wa,\
                        weightua=self._ua, weightva=self._va)

        if self.scaling_factor is not None:
            score = score/tf.sqrt(self.scaling_factor)

        if mask_enc is not None:
            score = score + (tf.cast(tf.math.equal(mask_enc, False), score.dtype)*-1e9)

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

    def get_config(self):
        '''Config'''
        config = {'units': self.units,
                  'probability_fn': serialize(self.probability_fn),
                  'dropout_rate' : self.dropout_rate,
                  'return_aweights' : self.return_aweights,
                  'scaling_factor' : self.scaling_factor,
                  'weights_initializer' : initializers.serialize(self.weights_initializer),
                  'bias_initializer' : initializers.serialize(self.bias_initializer),
                  'weights_regularizer' : regularizers.serialize(self.weights_regularizer),
                  'bias_regularizer' : regularizers.serialize(self.bias_regularizer),
                  'weights_constraint' : constraints.serialize(self.weights_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(BahdanauAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LuongeAttention(Layer):
    '''
    LuongeAttention
    Implemented based on below paper
    https://arxiv.org/pdf/1508.04025.pdf
    # Arguments
        units = number of hidden units to use.
        attention_type = Type of attention, it takes any of 'dot', 'general', 'concat'
        probability_fn = probability function to get probabilities(weights for attention)
                         You can use 'softmax' or 'hardmax' or 'sparsemax' or any custom
                         function which takes input distribution and returns probability dist.
        dropout_rate = dropout for attention weights (between 0 and 1, 0 - no dropout).
        return_aweights = Bool, whether to return attention weights or not.
        scaling_factor = int/float to scale the score vector. default None=1
        weights_initializer = initializer for weight matrix
        weights_regularizer = Regularize the weights (W, V)
        weights_constraint = Constraint function applied to the weights
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
    # Raises:
        ValueError: if attention type is not one of 'dot', 'general', 'concat'.
    '''
    def __init__(self, units,
                 attention_type='dot',
                 probability_fn='softmax',
                 dropout_rate=0,
                 return_aweights=False,
                 scaling_factor=None,
                 weights_initializer='he_normal',
                 weights_regularizer=None,
                 weights_constraint=None, **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = ""
        super(LuongeAttention, self).__init__(**kwargs)

        self.units = units
        self.attention_type = attention_type
        self.probability_fn = cget(probability_fn)
        self.dropout_rate = dropout_rate
        self.return_aweights = return_aweights
        self.scaling_factor = scaling_factor
        self.weights_initializer = initializers.get(weights_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)

        if self.attention_type == 'general':
            self._wa = layers.Dense(self.units, use_bias=False,\
                kernel_initializer=weights_initializer, kernel_regularizer=weights_regularizer,\
                    kernel_constraint=weights_constraint,\
                        name=self.name+"Wa")
        elif self.attention_type == 'concat':
            self._wa = layers.Dense(self.units, use_bias=False,\
                kernel_initializer=weights_initializer, kernel_regularizer=weights_regularizer,\
                    kernel_constraint=weights_constraint,\
                        name=self.name+"Wa")
            self._va = layers.Dense(1, use_bias=False, kernel_initializer=self.weights_initializer,\
                kernel_regularizer=weights_regularizer, kernel_constraint=self.weights_constraint,\
                    name=self.name+"Va")
        self.supports_masking = True


    def build(self, input_shape):
        '''build'''
        assert isinstance(input_shape, dict)

        shape_en, shape_dc = input_shape['enocderHs'], input_shape['decoderHt']

        assert len(shape_en) >= 3, "Encoder Hiddenstates/output should be 3 dim \
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

        enc_out, dec_ht = tf.cast(inputs['enocderHs'], tf.float32), \
            tf.cast(inputs['decoderHt'], tf.float32)

        if mask_dec is not None:
            dec_ht = dec_ht * tf.cast(mask_dec, dec_ht.dtype)
        if mask_enc is not None:
            enc_out = enc_out * tf.cast(tf.expand_dims(mask_enc, 2), enc_out.dtype)

        dec_ht_with_tax = tf.expand_dims(dec_ht, 1)

        #score shape (batch_size, max_length)
        if self.attention_type == 'dot':
            score = _attention_score(dec_ht=dec_ht_with_tax, enc_hs=enc_out,\
                        attention_type='dot')
        elif self.attention_type == 'general':
            score = _attention_score(dec_ht=dec_ht_with_tax, enc_hs=enc_out,\
                        attention_type='general', weightwa=self._wa)
        elif self.attention_type == 'concat':
            score = _attention_score(dec_ht=dec_ht_with_tax, enc_hs=enc_out,\
                    attention_type='concat', weightwa=self._wa, weightva=self._va)
        else:
            raise ValueError("mode must be 'dot', 'general', or 'concat'.")

        if self.scaling_factor is not None:
            score = score/tf.sqrt(self.scaling_factor)

        if mask_enc is not None:
            score = score + (tf.cast(tf.math.equal(mask_enc, False), score.dtype)*-1e9)

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

    def get_config(self):
        '''Config'''
        config = {'units': self.units,
                  'attention_type': self.attention_type,
                  'probability_fn': serialize(self.probability_fn),
                  'dropout_rate' : self.dropout_rate,
                  'return_aweights' : self.return_aweights,
                  'scaling_factor' : self.scaling_factor,
                  'weights_initializer' : initializers.serialize(self.weights_initializer),
                  'weights_regularizer' : regularizers.serialize(self.weights_regularizer),
                  'weights_constraint' : constraints.serialize(self.weights_constraint)}
        base_config = super(LuongeAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _monotonic_attetion(probabilities, attention_prev, mode):

    """Compute monotonic attention distribution from choosing probabilities.

    Implemented Based on -
    https://colinraffel.com/blog/online-and-linear-time-attention-by-enforcing-monotonic-alignments.html
    https://arxiv.org/pdf/1704.00784.pdf
    Mainly implemented by referring
    https://github.com/craffel/mad/blob/b3687a70615044359c8acc440e43a5e23dc58309/example_decoder.py#L22

    # Arguments:
        probabilities: Probability of choosing input sequence..
                       Should be of shape (batch_size, max_length),
                       and should all be in the range [0, 1].
        previous_attention: The attention distribution from the previous output timestep.
                            Should be of shape (batch_size, max_length).
                            For the first output timestep,
                            should be [1, 0, 0, ...,0] for all n in [0, ... batch_size - 1].
        mode: How to compute the attention distribution.
              Must be one of 'recursive', 'parallel', or 'hard'.

              - 'recursive' uses tf.scan to recursively compute the distribution.
              This is slowest but is exact, general, and does not suffer from
              numerical instabilities.

              - 'parallel' uses parallelized cumulative-sum and cumulative-product
              operations to compute a closed-form solution to the recurrence relation
              defining the attention distribution.  This makes it more efficient than 'recursive',
              but it requires numerical checks which make the distribution non-exact.
              This can be a problem in particular when max_length is long and/or
              probabilities has entries very close to 0 or 1.

              - 'hard' requires that  the probabilities in p_choose_i are all either 0 or 1,
              and subsequently uses a more efficient and exact solution.
    # Returns: A tensor of shape (batch_size, max_length) representing the attention distributions
               for each sequence in the batch.

    # Raises:
             ValueError: if mode is not one of 'recursive', 'parallel', 'hard'."""
    if mode == 'hard':
        #Remove any probabilities before the index chosen last time step
        probabilities = probabilities*tf.cumsum(attention_prev, axis=1)
        attention = probabilities*tf.cumprod(1-probabilities, axis=1, exclusive=True)
    elif mode == 'recursive':
        batch_size = tf.shape(probabilities)[0]
        shifted_1mp_probabilities = tf.concat([tf.ones((batch_size, 1)),\
            1 - probabilities[:, :-1]], 1)
        attention = probabilities*tf.transpose(tf.scan(lambda x, yz: tf.reshape(yz[0]*x + yz[1],\
            (batch_size,)), [tf.transpose(shifted_1mp_probabilities),\
                tf.transpose(attention_prev)], tf.zeros((batch_size,))))
    elif mode == 'parallel':
        cumprod_1mp_probabilities = tf.exp(tf.cumsum(tf.math.log(tf.clip_by_value(1-probabilities,\
            1e-10, 1)), axis=1, exclusive=True))
        attention = probabilities*cumprod_1mp_probabilities*tf.cumsum(attention_prev/\
            tf.clip_by_value(cumprod_1mp_probabilities, 1e-10, 1.), axis=1)
    else:
        raise ValueError("Mode must be 'hard', 'parallel' or 'recursive' ")

    return attention

class MonotonicBahdanauAttention(Layer):
    '''
    MonotonicBahdanauAttention
    '''
    def __init__(self, units,
                 mode='parallel',
                 return_aweights=False,
                 scaling_factor=None,
                 noise_std=0,
                 weights_initializer='he_normal',
                 bias_initializer='zeros',
                 weights_regularizer=None,
                 bias_regularizer=None,
                 weights_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = ""
        super(MonotonicBahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.return_aweights = return_aweights
        self.scaling_factor = scaling_factor
        self.noise_std = noise_std
        self.weights_initializer = initializers.get(weights_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.weights_regularizer = regularizers.get(bias_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self._wa = layers.Dense(self.units, use_bias=False,\
            kernel_initializer=weights_initializer, bias_initializer=bias_initializer,\
                kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                    kernel_constraint=weights_constraint, bias_constraint=bias_constraint,\
                        name=self.name+"Wa")
        self._ua = layers.Dense(self.units,\
            kernel_initializer=weights_initializer, bias_initializer=bias_initializer,\
                kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                    kernel_constraint=weights_constraint, bias_constraint=bias_constraint,\
                        name=self.name+"Ua")
        self._va = layers.Dense(1, use_bias=False, kernel_initializer=weights_initializer,\
            kernel_regularizer=weights_regularizer, bias_regularizer=bias_regularizer,\
                bias_initializer=bias_initializer, kernel_constraint=weights_constraint,\
                    bias_constraint=bias_constraint, name=self.name+"Va")

    def build(self, input_shape):
        '''build'''
        assert isinstance(input_shape, dict)

        shape_en, shape_dc = input_shape['enocderHs'], input_shape['decoderHt']

        assert len(shape_en) >= 3, "Encoder Hiddenstates/output should be 3 dim or more \
        ( B x T x H ), but got {} dim".format(len(shape_en))

        assert len(shape_dc) == 2, "Decoder Hidden/output should be 2 \
        dim (B x H), but got {} dim".format(len(shape_dc))

        self.built = True # pylint: disable=W0201

    def call(self, inputs, mask=None, training=True):
        '''call'''
        assert isinstance(inputs, dict)

        if ('enocderHs' not in inputs.keys())or ('decoderHt' not in inputs.keys()\
            or 'prevAttention' not in inputs.keys()):
            raise ValueError("Input to the layer must be a dict with \
            keys=['enocderHs','decoderHt', 'prevAttention']")

        if isinstance(mask, dict):
            mask_enc = mask.get('enocderHs', None)
            mask_dec = mask.get('decoderHt', None)
        else:
            mask_enc = mask
            mask_dec = None
        enc_out, dec_prev_hs = tf.cast(inputs['enocderHs'], tf.float32), \
            tf.cast(inputs['decoderHt'], tf.float32)

        prev_attention = inputs['prevAttention']

        if mask_dec is not None:
            dec_prev_hs = dec_prev_hs * tf.cast(mask_dec, dec_prev_hs.dtype)
        if mask_enc is not None:
            enc_out = enc_out * tf.cast(tf.expand_dims(mask_enc, 2), enc_out.dtype)

        # decprev_hs - Decoder hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        dec_hidden_with_time_axis = tf.expand_dims(dec_prev_hs, 1)

        # score shape == (batch_size, max_length)
        score = _attention_score(dec_ht=dec_hidden_with_time_axis, enc_hs=enc_out,\
                    attention_type='bahdanau', weightwa=self._wa,\
                        weightua=self._ua, weightva=self._va)

        if self.scaling_factor is not None:
            score = score/tf.sqrt(self.scaling_factor)

        if training:
            if self.noise_std > 0:
                random_noise = tf.random.normal(shape=tf.shape(score), mean=0,\
                    stddev=self.noise_std, dtype=score.dtype, seed=self.seed)
                score = score + random_noise
        if self.mode == 'hard':
            probabilities = tf.cast(score > 0, score.dtype)
        else:
            probabilities = tf.sigmoid(score)

        attention_weights = _monotonic_attetion(probabilities, prev_attention, self.mode)
        attention_weights = tf.expand_dims(attention_weights, 1)

        #context_vector shape (batch_size, hidden_size)
        context_vector = tf.matmul(attention_weights, enc_out)
        context_vector = tf.squeeze(context_vector, 1, name="context_vector")

        if self.return_aweights:
            return context_vector, tf.squeeze(attention_weights, 1, name='attention_weights')
        return context_vector
