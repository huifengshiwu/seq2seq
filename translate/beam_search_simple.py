import tensorflow as tf
from translate import utils


"""
Code from: https://github.com/vahidk/EffectiveTensorflow
"""

def get_weights(sequence, eos_id, include_first_eos=True):
    cumsum = tf.cumsum(tf.to_float(tf.not_equal(sequence, eos_id)), axis=1)
    range_ = tf.range(start=1, limit=tf.shape(sequence)[1] + 1)
    range_ = tf.tile(tf.expand_dims(range_, axis=0), [tf.shape(sequence)[0], 1])
    weights = tf.to_float(tf.equal(cumsum, tf.to_float(range_)))

    if include_first_eos:
        weights = weights[:,:-1]
        shape = [tf.shape(weights)[0], 1]
        weights = tf.concat([tf.ones(tf.stack(shape)), weights], axis=1)

    return tf.stop_gradient(weights)


def resize_like(src, dst):
    batch_size = tf.shape(src)[0]
    beam_size = tf.shape(dst)[0] // batch_size
    shape = get_shape(src)[1:]
    src = tf.tile(tf.expand_dims(src, axis=1), [1, beam_size] + [1] * len(shape))
    src = tf.reshape(src, tf.stack([batch_size * beam_size] + shape))
    return src


def get_shape(tensor):
    """Returns static shape if available and dynamic shape otherwise."""
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def batch_gather(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseduocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = get_shape(tensor)
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output


def log_softmax(x, axis, temperature=None):
    T = temperature or 1.0
    my_max = tf.reduce_max(x/T, axis=axis, keep_dims=True)
    return x - (tf.log(tf.reduce_sum(tf.exp(x/T - my_max), axis, keep_dims=True)) + my_max)


def rnn_beam_search(update_funs, initial_states, sequence_length, beam_width, len_normalization=None,
                    early_stopping=True, temperature=None):
    """
    :param update_funs: function to compute the next state and logits given the current state and previous ids
    :param initial_states: recurrent model states
    :param sequence_length: maximum output length
    :param beam_width: beam size
    :param len_normalization: length normalization coefficient (0 or None for no length normalization)
    :param early_stopping: same behavior as the old beam-search implementation, beam size is reduced by one
        each time a new finished hypothesis is encountered. This doesn't improve decoding speed, but
        might produce better results.
        FIXME: this takes a lot of memory. This causes crashes with very long sequences
    :return: tensor of size (batch_size, beam_size, seq_len) containing the beam-search hypotheses sorted by
        best score (axis 1), and tensor of size (batch_size, beam_size) containing the said scores.
    """
    batch_size = tf.shape(initial_states[0])[0]

    state = tf.tile(tf.expand_dims(initial_states[0], axis=1), [1, beam_width, 1])

    sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])
    sel_sum_logprobs = tf.tile(sel_sum_logprobs, [batch_size, 1])

    ids = tf.tile([[utils.BOS_ID]], [batch_size, beam_width])
    sel_ids = tf.expand_dims(ids, axis=2)

    mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

    for i in range(sequence_length):
        ids = tf.reshape(ids, [batch_size * beam_width])

        state = tf.reshape(state, [batch_size * beam_width, tf.shape(state)[2]])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=True):
            state, logits = update_funs[0](state, ids, i)

        state = tf.reshape(state, [batch_size, beam_width, tf.shape(state)[1]])
        num_classes = logits.shape.as_list()[-1]
        logits = tf.reshape(logits, [batch_size, beam_width, num_classes])
        logits = log_softmax(logits, axis=2, temperature=temperature)

        sum_logprobs = (tf.expand_dims(sel_sum_logprobs, axis=2) + (logits * tf.expand_dims(mask, axis=2)))

        num_classes = logits.shape.as_list()[-1]
        sel_sum_logprobs, indices = tf.nn.top_k(
            tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
            k=beam_width)

        ids = indices % num_classes

        beam_ids = indices // num_classes
        state = batch_gather(state, beam_ids)
        sel_ids = tf.concat([batch_gather(sel_ids, beam_ids), tf.expand_dims(ids, axis=2)], axis=2)
        mask = (batch_gather(mask, beam_ids) * tf.to_float(tf.not_equal(ids, utils.EOS_ID)))


    sel_ids = sel_ids[:, :, 1:]  # remove BOS symbol

    return sel_ids, sel_sum_logprobs
