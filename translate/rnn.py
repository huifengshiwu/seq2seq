from tensorflow.python.ops import init_ops
import tensorflow as tf


def stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, initial_states_fw=None, initial_states_bw=None,
                                    dtype=None, sequence_length=None, parallel_iterations=None, scope=None,
                                    time_pooling=None, pooling_avg=None, initializer=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with tf.variable_scope(scope or "stack_bidirectional_rnn", initializer=initializer):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with tf.variable_scope('cell_{}'.format(i)):
                outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype)
                # Concat the outputs to create the new input.
                prev_layer = tf.concat(outputs, axis=2)

                if time_pooling and i < len(cells_fw) - 1:
                    prev_layer, sequence_length = apply_time_pooling(prev_layer, sequence_length, time_pooling[i],
                                                                     pooling_avg)

            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def apply_time_pooling(inputs, sequence_length, stride, pooling_avg=False):
    shape = [tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2].value]

    if pooling_avg:
        inputs_ = [inputs[:, i::stride, :] for i in range(stride)]

        max_len = tf.shape(inputs_[0])[1]
        for k in range(1, stride):
            len_ = tf.shape(inputs_[k])[1]
            paddings = tf.stack([[0, 0], [0, max_len - len_], [0, 0]])
            inputs_[k] = tf.pad(inputs_[k], paddings=paddings)

        inputs = tf.reduce_sum(inputs_, axis=0) / len(inputs_)
    else:
        inputs = inputs[:, ::stride, :]

    inputs = tf.reshape(inputs, tf.stack([shape[0], tf.shape(inputs)[1], shape[2]]))
    sequence_length = (sequence_length + stride - 1) // stride  # rounding up

    return inputs, sequence_length


class CellInitializer(init_ops.Initializer):
    """
    Orthogonal initialization of recurrent connections, like in Bahdanau et al. 2015
    """
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.default_initializer = tf.get_variable_scope().initializer or init_ops.glorot_uniform_initializer()
        self.initializer = tf.orthogonal_initializer()

    def __call__(self, shape, dtype=None, partition_info=None, verify_shape=None):
        assert shape[1] % self.cell_size == 0
        input_size = shape[0] - self.cell_size

        W, U = [], []
        for _ in range(shape[1] // self.cell_size):
            W.append(self.default_initializer(shape=[input_size, self.cell_size]))
            U.append(self.initializer(shape=[self.cell_size, self.cell_size]))

        return tf.concat([tf.concat(W, axis=1), tf.concat(U, axis=1)], axis=0)


class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None,
                 layer_norm=False):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state, scope=None):
        inputs = tf.concat(inputs, axis=1)

        with tf.variable_scope("gates"):
            bias_ones = self._bias_initializer
            dtype = [a.dtype for a in [inputs, state]][0]
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)

            bias = tf.get_variable('bias', [2 * self._num_units], dtype=dtype, initializer=bias_ones)
            value = _linear([inputs, state], 2 * self._num_units, bias=False,
                            kernel_initializer=self._kernel_initializer)

            r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

            if self._layer_norm:
                r = tf.contrib.layers.layer_norm(r, scope='reset', center=False)
                u = tf.contrib.layers.layer_norm(u, scope='update', center=False)

            rb, ub = tf.split(bias, num_or_size_splits=2)
            r = tf.nn.sigmoid(r + rb)
            u = tf.nn.sigmoid(u + ub)

        with tf.variable_scope("candidate"):
            bias = tf.get_variable('bias', [self._num_units], dtype=dtype, initializer=self._bias_initializer)
            c = _linear([inputs, r * state], self._num_units, bias=False, kernel_initializer=self._kernel_initializer)

            if self._layer_norm:
                c = tf.contrib.layers.layer_norm(c, center=False)

            c = self._activation(c + bias)

        new_h = u * state + (1 - u) * c
        return new_h, new_h


def _linear(args, output_size, bias, bias_initializer=None, kernel_initializer=None):
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable('kernel', [total_arg_size, output_size], dtype=dtype, initializer=kernel_initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable('bias', [output_size], dtype=dtype, initializer=bias_initializer)
        return tf.nn.bias_add(res, biases)
