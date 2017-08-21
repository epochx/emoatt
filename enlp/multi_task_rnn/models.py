#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from .encoders import encoder_RNN_embed
from .decoders import (seq_regression_decoder_linear, seq_classification_decoder_linear,
                       seq_regression_decoder_attention)

def dep_attention_RNN_linear(encoder_inputs,
                             encoder_extra_inputs,
                             cell,
                             num_encoder_symbols,
                             num_decoder_symbols,
                             word_embedding_size,
                             batch_size,
                             task,
                             decoder_inputs=None,
                             binary_feat_dim=0,
                             class_dim=1,
                             sequence_length=None,
                             loop_function=None,
                             context_win_size=1,
                             train_embeddings=True,
                             backwards_cell=None,
                             dtype=dtypes.float32):
  """
  Encoder and decoder share weights.
  """
  enc_outputs = encoder_RNN_embed(cell, encoder_inputs, num_encoder_symbols, word_embedding_size,
                                  sequence_length, batch_size, context_win_size=context_win_size,
                                  train_embeddings=train_embeddings, dtype=dtype,
                                  backwards_cell=backwards_cell)
  # num_decoder_symbols=num_labeling_decoder_symbols,

  encoder_embedded_inputs, encoder_outputs, encoder_state = enc_outputs

  if binary_feat_dim:
    #print(encoder_inputs[0].get_shape())
    #print(encoder_outputs[0].get_shape())
    # set shape for extra inputs
    [tnsr.set_shape((batch_size, binary_feat_dim)) for tnsr in encoder_extra_inputs]
    #print(encoder_extra_inputs[0].get_shape())
    #print(type(encoder_extra_inputs[0]))
    extended_encoder_outputs = []
    for i in xrange(len(encoder_outputs)):
      extended_encoder_outputs.append(array_ops.concat([encoder_outputs[i],
                                                        encoder_extra_inputs[i]],
                                                       1))
  else:
    extended_encoder_outputs = encoder_outputs

  if task["regression"]:
    #seq_regression_decoder_linear

    reg_dec_outputs = seq_regression_decoder_attention(extended_encoder_outputs,
                                                       encoder_state, batch_size,
                                                       decoder_inputs=decoder_inputs,
                                                       class_dim=class_dim,
                                                       sequence_length=sequence_length)
    reg_decoder_outputs, reg_attention_weights = reg_dec_outputs
  else:
    reg_decoder_outputs, reg_attention_weights = None, None

  if task['classification']:
    class_dec_outputs = seq_classification_decoder_linear(num_decoder_symbols,
                                                          extended_encoder_outputs,
                                                          encoder_state, batch_size)

    class_decoder_outputs, class_attention_weights =  class_dec_outputs
  else:
    class_decoder_outputs, class_attention_weights = None, None

  return reg_decoder_outputs, reg_attention_weights, \
         class_decoder_outputs, class_attention_weights
