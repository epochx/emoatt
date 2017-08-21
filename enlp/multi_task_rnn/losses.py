#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_pearson_correlation


def get_classification_loss(logits, targets, softmax_loss_function=None):
  bucket_outputs = logits
  if softmax_loss_function is None:
    assert len(bucket_outputs) == len(targets) == 1
    # We need to make target an int64-tensor and set its shape.
    bucket_target = array_ops.reshape(math_ops.to_int64(targets[0]), [-1])
    crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(bucket_outputs[0], bucket_target)
  else:
    assert len(bucket_outputs) == len(targets) == 1
    crossent = softmax_loss_function(bucket_outputs[0], targets[0])

  batch_size = array_ops.shape(targets[0])[0]
  loss = tf.reduce_sum(crossent) / math_ops.cast(batch_size, dtypes.float32)

  return loss



def get_regression_squared_loss(predicts, targets):
    assert len(predicts) == len(targets) == 1
    sqared_error = tf.nn.l2_loss(math_ops.to_float(predicts)-math_ops.to_float(targets))
    batch_size = array_ops.shape(targets[0])[0]
    loss = tf.reduce_sum(sqared_error) / math_ops.cast(batch_size, dtypes.float32)
    return loss


def get_regression_pearson_loss(predicts, targets):
  assert len(predicts) == len(targets) == 1

  flat_predicts = tf.reshape(predicts, [-1])
  flat_targets = tf.reshape(targets, [-1])

  loss = tf.negative(PearsonCorrelationTF(flat_targets, flat_predicts))
  return loss


# Use TF to compute the Pearson Correlation of a pair of 1-dimensional vectors.
# From: https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
def PearsonCorrelationTF(x, y, prefix='pearson'):
  '''Create a TF network that calculates the Pearson Correlation on two input
  vectors.  Returns a scalar tensor with the correlation [-1:1].'''
  with tf.name_scope(prefix):
    n = tf.to_float(tf.shape(x)[0])
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)
    xy_sum = tf.reduce_sum(tf.multiply(x, y))
    x2_sum = tf.reduce_sum(tf.multiply(x, x))
    y2_sum = tf.reduce_sum(tf.multiply(y, y))

    r_num = tf.subtract(tf.multiply(n, xy_sum), tf.multiply(x_sum, y_sum))
    r_den_x = tf.sqrt(tf.subtract(tf.multiply(n, x2_sum), tf.multiply(x_sum, x_sum)))
    r_den_y = tf.sqrt(tf.subtract(tf.multiply(n, y2_sum), tf.multiply(y_sum, y_sum)))
    r = tf.div(r_num, tf.multiply(r_den_x, r_den_y), name='r')
  return r