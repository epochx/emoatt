#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attentions import  get_vinyals_attention_function
from .utils import _linear
import tensorflow as tf
from tensorflow.python.ops import array_ops



def seq_classification_decoder_linear(num_decoder_symbols, encoder_outputs,
                                      att_initial_state, batch_size,
                                      sequence_length=None, scope=None):
  """

  Simple linear decoder,


  :param num_decoder_symbols:
  :param decoder_inputs:
  :param loop_function:
  :param sequence_length:
  :param scope:
  :return:
  """
  with tf.variable_scope("seq_classification_decoder_linear", reuse=None) as scope:
    output_size = encoder_outputs[0].get_shape()[1].value
    attention = get_vinyals_attention_function(encoder_outputs, output_size, 1, scope=scope)
    attn_weights, attns = attention(att_initial_state)

    # with variable_scope.variable_scope(scope or "Linear"):
    output = _linear(attns[0], num_decoder_symbols, True)

    return [output], attn_weights


def seq_regression_decoder_attention(encoder_outputs,  att_initial_state, batch_size,
                                  sequence_length=None, scope=None,
                                  class_dim=1, decoder_inputs=None):
  """

  Simple linear decoder,


  :param num_decoder_symbols:
  :param decoder_inputs:
  :param loop_function:
  :param sequence_length:
  :param scope:
  :return:
  """
  with tf.variable_scope("seq_regression_decoder_linear", reuse=None) as scope:
    output_size = encoder_outputs[0].get_shape()[1].value
    attention = get_vinyals_attention_function(encoder_outputs, output_size, 1, scope=scope)
    attn_weights, attns = attention(att_initial_state)
    regression_input = attns[0]
    #print(regression_input)
    #print(decoder_inputs)

    if decoder_inputs is not None and class_dim > 1:

      one_hot_decoder_inputs =  [tf.one_hot(decoder_input, class_dim, dtype=tf.float32)
                                 for decoder_input in decoder_inputs]

      concat_decoder_inputs = tf.concat(one_hot_decoder_inputs, 1)
      concat_decoder_inputs.set_shape([None, class_dim])

      #print(concat_decoder_inputs)
      extended_regression_input = tf.concat([regression_input,
                                             concat_decoder_inputs],
                                            1)

    else:
      extended_regression_input = regression_input

    #print(extended_regression_input)
    #import ipdb; ipdb.set_trace()
    with tf.variable_scope("Linear_1") as scope:
      output_1= tf.sigmoid(_linear(extended_regression_input, 200, True, scope=scope))

    #with tf.variable_scope("Linear_2") as scope:
    #  output_1= tf.sigmoid(_linear(output_1, 100, True, scope=scope))

    with tf.variable_scope("Linear_3") as scope:
      output_1= tf.sigmoid(_linear(output_1, 50, True, scope=scope))

    #with tf.variable_scope("Linear_4") as scope:
    #  output_1= tf.sigmoid(_linear(output_1, 25, True, scope=scope))

    with tf.variable_scope("Linear_5") as scope:
      output = _linear(output_1, 1, True, scope=scope)

    return [output], attn_weights

def seq_regression_decoder_linear(encoder_outputs,  att_initial_state, batch_size,
                                  sequence_length=None, scope=None,
                                  class_dim=1, decoder_inputs=None):
  """

  Simple linear decoder,


  :param num_decoder_symbols:
  :param decoder_inputs:
  :param loop_function:
  :param sequence_length:
  :param scope:
  :return:
  """
  with tf.variable_scope("seq_regression_decoder_simple", reuse=None) as scope:

    regression_input = att_initial_state

    if decoder_inputs is not None and class_dim > 1:

      one_hot_decoder_inputs =  [tf.one_hot(decoder_input, class_dim, dtype=tf.float32)
                                 for decoder_input in decoder_inputs]

      concat_decoder_inputs = tf.concat(one_hot_decoder_inputs, 1)
      concat_decoder_inputs.set_shape([None, class_dim])

      #print(concat_decoder_inputs)
      extended_regression_input = tf.concat([regression_input,
                                             concat_decoder_inputs],
                                            1)

    else:
      extended_regression_input = regression_input

    #print(extended_regression_input)
    #import ipdb; ipdb.set_trace()
    with tf.variable_scope("Linear_1") as scope:
      output_1= tf.sigmoid(_linear(extended_regression_input, 200, True, scope=scope))

    with tf.variable_scope("Linear_2") as scope:
      output_1= tf.sigmoid(_linear(output_1, 100, True, scope=scope))

    with tf.variable_scope("Linear_3") as scope:
      output_1= tf.sigmoid(_linear(output_1, 50, True, scope=scope))

    with tf.variable_scope("Linear_4") as scope:
      output_1= tf.sigmoid(_linear(output_1, 25, True, scope=scope))

    with tf.variable_scope("Linear_5") as scope:
      output = tf.sigmoid(tf.sigmoid(_linear(output_1, 1, True, scope=scope)))

    return [output], None