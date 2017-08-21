#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import subprocess
import sys
import time
import hashlib
import argparse
import copy

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from enlp.eval import classeval, regeval
from enlp.multi_task_rnn import MultiTaskModel



def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  if v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("--learning_rate",
                      type=float,
                      default=0.5,
                      help="(initial) Learning rate.")

  parser.add_argument("--decay_factor",
                      type=float,
                      default=0.9,
                      help="Learning rate decay factor")

  parser.add_argument("--decay_end",
                      type=float,
                      default=5000,
                      help="Learning rate decay final step, when using sgd")

  parser.add_argument("--early_stopping",
                      type=int,
                      default=2000,
                      help="Stop training if no improvement")

  parser.add_argument("--optimizer",
                      type=str,
                      default="sgd",
                      choices=["adam", "sgd"],
                      help="Optimizer")

  parser.add_argument("--max_gradient_norm",
                      type=float,
                      default=5.0,
                      help="Clip gradients to this norm.")

  parser.add_argument("--batch_size",
                      type=int,
                      default=8,
                      help="Batch size to use during training.")

  parser.add_argument("--eval_batch_size",
                      type=int,
                      default=1,
                      help="Batch size to use during evaluation.")

  parser.add_argument("--size",
                      default=100,
                      type=int,
                      help="Size of each model layer.")

  parser.add_argument("--word_embedding_size",
                      default=0,
                      type=int,
                      help="Size of the word embedding. Use 0 to use json data provided.")

  parser.add_argument("--train_word_embeddings",
                      default=True,
                      type=str2bool,
                      help="Train word embeddings")

  parser.add_argument("--num_layers",
                      default=1,
                      type=int,
                      help="Number of layers in the model.")

  parser.add_argument("--regularization_lambda",
                      default=0.2,
                      type=float,
                      help="Coefficient of L2 regularization (set 0 for no regularization).")

  parser.add_argument("--json_path",
                      required=True,
                      type=str,
                      help="JSON Data dir (for folds) or file path. Checks JSON_DIR first.")

  parser.add_argument("--results_path",
                      default="",
                      help="Path to save trained model and outputs. Checks RESULTS_DIR first")

  parser.add_argument("--checkpoint_path",
                      default="",
                      type=str,
                      help="Path to load trained model")

  parser.add_argument("--steps_per_checkpoint",
                      default=100,
                      type=int,
                      help="How many training steps to do per checkpoint.")

  parser.add_argument("--steps_per_eval",
                      default=100,
                      type=int,
                      help="How many training steps to do before evaluation.")

  parser.add_argument("--max_training_steps",
                      default=5000,
                      type=int,
                      help="Max training steps.")

  parser.add_argument("--use_attention",
                      default=True,
                      type=str2bool,
                      help="Use attention based RNN")

  parser.add_argument("--max_sequence_length",
                      default=50,
                      type=int,
                      help="Max sequence length.")

  parser.add_argument("--context_win_size",
                      default=3,
                      type=int,
                      help="Context window size.")

  parser.add_argument("--dropout_keep_prob",
                      default=0.9 ,
                      type=float,
                      help="dropout keep cell input and output prob.")

  parser.add_argument("--zoneout_keep_prob",
                      default=1,
                      type=float,
                      help="zoneout keep cell input and output prob.")

  parser.add_argument("--bidirectional_rnn",
                      default=True,
                      type=str2bool,
                      help="Use bidirectional RNN")

  parser.add_argument("--task",
                      default="regression",
                      type=str,
                      choices=["joint", "regression"],
                      help="Task")

  parser.add_argument("--use_binary_features",
                      default=True,
                      type=str2bool,
                      help="Use binary features")

  parser.add_argument("--overwrite",
                      default=False,
                      type=str2bool,
                      help="Rewrite trained model")

  parser.add_argument("--rnn",
                      default="lstm",
                      choices=["lstm", "gru", "bnlstm"],
                      help="Rnn cell type",
                      type=str)

  parser.add_argument("--loss",
                      default="pc",
                      choices=["mse", "pc"],
                      help="Loss function",
                      type=str)

  FLAGS = parser.parse_args()

  _buckets = [(FLAGS.max_sequence_length,)]


  if FLAGS.max_sequence_length == 0:
      print('Please indicate max sequence length. Exit')
      exit()

  if FLAGS.zoneout_keep_prob < 1 and FLAGS.dropout_keep_prob < 1:
      print('Please choose dropout ore zoneout. Exit')
      exit()

  if FLAGS.task is None:
      print('Please indicate task to run. Available options: intent; tagging; joint')
      exit()

  task = dict({'classification': 0, 'regression': 0, 'joint': 0})
  if FLAGS.task == 'classification':
      task['classification'] = 1
  elif FLAGS.task == 'regression':
      task['regression'] = 1
  elif FLAGS.task == 'joint':
      task['regression'] = 1
      task['classification'] = 1
      task['joint'] = 1

  FLAGS.task = task

  json_dir = os.environ.get('JSON_DIR', None)
  results_dir = os.environ.get('RESULTS_DIR', None)

  if json_dir:
    FLAGS.json_path = os.path.join(json_dir, FLAGS.json_path)

  if results_dir:
    FLAGS.results_path = os.path.join(results_dir, FLAGS.results_path)

  if os.path.isdir(FLAGS.json_path):
    if FLAGS.json_path.endswith("/"):
      FLAGS.json_path = FLAGS.json_path[:-1]
    json_base_path, json_path = os.path.split(FLAGS.json_path)

    paths = [(json_base_path, json_path, json_filename)
             for json_filename in sorted(os.listdir(FLAGS.json_path))]

  elif os.path.isfile(FLAGS.json_path):
    json_base_dir, json_filename = os.path.split(FLAGS.json_path)
    paths = [(json_base_dir, "", json_filename) ]
  else:
    print("Not valid JSON path")
    exit()

  BASE_FLAGS=FLAGS

  base_results_path = FLAGS.results_path

  for json_base_path, json_path, json_filename in paths:

    FLAGS = copy.copy(FLAGS)
    # we obtain model_id before modifying paths, so the same model keeps the id for different folds
    model_id = get_model_id(FLAGS)

    # if folds this is json_path/
    FLAGS.json_path = os.path.join(*(json_base_path,
                                     json_path,
                                     json_filename))

    FLAGS.results_path = os.path.join(*(base_results_path,
                                        json_path,
                                        json_filename))

    if not os.path.isdir(FLAGS.results_path):
      os.makedirs(FLAGS.results_path)

    print('Applying Parameters:')
    for k, v in vars(FLAGS).iteritems():
      print('%s: %s' % (k, str(v)))

    results_dir = os.path.join(FLAGS.results_path, model_id)

    if not os.path.isdir(results_dir):
      os.makedirs(results_dir)
    else:
      if FLAGS.overwrite:
        print("Model already trained, overwriting")
      else:
        print("Model already trained. Set overwrite=True to overwrite\n")
        continue

    # store parameters
    with open(os.path.join(results_dir, "params.json"), "w") as f:
      json.dump(vars(FLAGS), f)

    log = []

    class_valid_out_file = os.path.join(results_dir, 'classification.valid.hyp.txt')
    class_test_out_file = os.path.join(results_dir, 'classification.test.hyp.txt')

    reg_valid_out_file = os.path.join(results_dir, 'regression.valid.txt')
    reg_test_out_file = os.path.join(results_dir, 'regression.test.txt')

    att_valid_out_file = os.path.join(results_dir, 'attentions.valid.hyp.json')
    att_test_out_file = os.path.join(results_dir, 'attentions.test.hyp.json')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      # Read data into buckets and compute their sizes.
      print("Reading train/valid/test data")
      train, valid, test, vocabs, rev_vocabs, binary_feat_dim, class_dim, embeddings_matrix = read_data(FLAGS)

      train_x, train_feat_x, train_y, train_z, train_ids = train
      valid_x, valid_feat_x, valid_y, valid_z, valid_ids = valid
      test_x, test_feat_x, test_y, test_z, test_ids = test
      vocab, label_vocab = vocabs
      rev_vocab, rev_label_vocab = rev_vocabs

      print(len(train_x))
      print(len(valid_x))
      print(len(test_x))

      train_set = generate_buckets(_buckets, train_x, train_y, train_z,
                                   train_ids, extra_x=train_feat_x)
      dev_set = generate_buckets(_buckets, valid_x, valid_y, valid_z,
                                 valid_ids, extra_x=valid_feat_x)
      test_set = generate_buckets(_buckets, test_x, test_y, test_z,
                                  test_ids, extra_x=test_feat_x)

      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))

      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                             for i in xrange(len(train_bucket_sizes))]

      # Create model.
      print("Max sequence length: %d." % _buckets[0][0])
      print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

      model, model_test = create_model(sess, len(vocab), len(label_vocab),
                                       embeddings_matrix, binary_feat_dim,
                                       class_dim, FLAGS, _buckets)

      # train_writer = tf.summary.FileWriter(results_dir, sess.graph)
      print("Creating model with source_vocab_size=%d and label_vocab_size=%d." %
            (len(vocab), len(label_vocab)))

      # This is the training loop.
      step_time, loss = 0.0, 0.0
      current_step = 0
      best_step = 0

      best_valid_regression_result = 0
      best_test_regression_result = 0
      best_valid_class_score = 0
      best_test_class_score = 0

      while model.global_step.eval() < FLAGS.max_training_steps:
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, _, encoder_extra_inputs, batch_sequence_length, values, labels, ids = \
            model.get_batch(train_set, bucket_id)

        if task['joint'] == 1:
          output = model.joint_step(sess, encoder_inputs, encoder_extra_inputs, values, labels,
                                    batch_sequence_length, bucket_id, False)
          _, step_loss, tagging_logits, tagging_att, class_logits, class_att = output

        elif task['regression'] == 1:
          output = model.regression_step(sess, encoder_inputs, encoder_extra_inputs, values, labels,
                                         batch_sequence_length, bucket_id, False)
          _, step_loss, tagging_logits, tagging_att = output
        elif task['classification'] == 1:
          output = model.classification_step(sess, encoder_inputs, encoder_extra_inputs,
                                             labels, batch_sequence_length, bucket_id, False)
          _, step_loss, class_logits, class_att = output

        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          print("global step %d step-time %.2f. Training loss %.2f"
                % (model.global_step.eval(), step_time, loss))
          sys.stdout.flush()
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(results_dir, "model.ckpt")
          step_time, loss = 0.0, 0.0

        # Once in a while, we run evals
        if current_step % FLAGS.steps_per_eval == 0:
          # valid
          valid_regression_result, valid_class_result = \
            run_valid_test(model_test, sess, dev_set, 'Valid', task,
                           rev_vocab, rev_label_vocab, FLAGS, _buckets,
                           regression_out_file=reg_valid_out_file,
                           classification_out_file=reg_test_out_file,
                           attentions_out_file=att_valid_out_file)

          # use pearson correlation as validation metric
          if valid_regression_result[1] > best_valid_regression_result:
            best_valid_regression_result = valid_regression_result[1]
            subprocess.call(['mv', reg_valid_out_file + ".hyp", reg_valid_out_file + '.hyp.best'])

            if task['joint'] == 1 and valid_class_result["all"]["a"] > best_valid_class_score:
              best_valid_class_score = valid_class_result["all"]["a"]
              subprocess.call(['mv', class_valid_out_file, class_valid_out_file + '.best'])

            model.saver.save(sess, checkpoint_path)
            best_step = current_step
            print("  New Best!")
            sys.stdout.flush()
            # test
            test_regression_result, test_class_result \
              = run_valid_test(model_test, sess, test_set, 'Test', task,
                              rev_vocab, rev_label_vocab, FLAGS, _buckets,
                              regression_out_file=reg_test_out_file,
                              classification_out_file=class_test_out_file,
                               attentions_out_file=att_test_out_file)

          print("")
          sys.stdout.flush()
          log.append([map(float, valid_regression_result), valid_class_result])

          with open(os.path.join(results_dir, "log.json"), "w") as f:
              json.dump(log, f)

          if FLAGS.early_stopping and current_step - best_step > FLAGS.early_stopping:
              break

    tf.reset_default_graph()
    FLAGS = BASE_FLAGS



def get_model_id(FLAGS):
  params = vars(FLAGS)
  sha = hashlib.sha1(str(params)).hexdigest()
  return sha


def generate_buckets(_buckets, x, y, z, example_ids, extra_x=None):
  data_set = [[] for _ in _buckets]
  for i, (source, target, label, idx) in enumerate(zip(x, y, z, example_ids)):
    for bucket_id, (source_size,) in enumerate(_buckets):
      if len(source) < source_size:
        if extra_x:
            data_set[bucket_id].append([source, extra_x[i], target, label, idx])
        else:
            data_set[bucket_id].append([source, None, target, label, idx])
        break
  return data_set # 4 outputs in each unit: source_ids, source_matrix, target_ids, label_ids



def read_data(FLAGS):
  """
  Reads the json and extracts the coresponding data

  :param json_path:
  :param task:
  :param FLAGS:
  :return:
  """
  with open(FLAGS.json_path, "r") as f:

    jsondic = json.load(f)

    len_train_x = len(jsondic["train_x"])
    len_valid_x = len(jsondic["valid_x"])
    len_test_x = len(jsondic["test_x"])

    print("Train examples: " + str(len_train_x))
    print("Valid examples: " + str(len_valid_x))
    print("Test examples: " + str(len_test_x))

    if len_test_x % FLAGS.eval_batch_size:
      # print("Reseting eval_batch_size to 1")
      # FLAGS.eval_batch_size = 1
      test_limit = len_test_x - (len_test_x % FLAGS.eval_batch_size)
      print("Incompatible eval_batch_size for testing, reseting to " + str(test_limit))
    else:
      test_limit = len_test_x

    if len_valid_x % FLAGS.eval_batch_size:
      valid_limit = len_valid_x - (len_valid_x % FLAGS.eval_batch_size)
      print("Incompatible eval_batch_size for validation, reseting to " + str(valid_limit))
    else:
      valid_limit = len_valid_x

    train_x, train_y = jsondic["train_x"], jsondic["train_y"]
    valid_x, valid_y = jsondic["valid_x"][:valid_limit], jsondic["valid_y"][:valid_limit]
    test_x, test_y = jsondic["test_x"][:test_limit], jsondic["test_y"][:test_limit]

    train_ids = jsondic["train_ids"]
    valid_ids = jsondic["valid_ids"][:valid_limit]
    test_ids  = jsondic["test_ids"][:test_limit]

    class_dim = jsondic["classdim"]

    if FLAGS.task['classification'] == 1:
      train_z = jsondic["train_z"]
      valid_z = jsondic["valid_z"][:valid_limit]
      test_z = jsondic["test_z"][:test_limit]

      label_vocab = jsondic["class2idx"]
      rev_label_vocab = {idx: token for token, idx in label_vocab.items()}

    else:
      if class_dim > 1:
        train_z = jsondic["train_z"]
        valid_z = jsondic["valid_z"][:valid_limit]
        test_z = jsondic["test_z"][:test_limit]
        label_vocab = jsondic["class2idx"]
        rev_label_vocab = {idx: token for token, idx in label_vocab.items()}
      else:
        # generating "fake" labels
        train_z = [0] * len(train_x)
        valid_z = [0] * len(valid_x)
        test_z = [0] * len(test_x)
        label_vocab = jsondic["class2idx"]
        rev_label_vocab = {idx: token for token, idx in label_vocab.items()}

    if FLAGS.word_embedding_size == 0:
      if "embeddings" in jsondic:
        print("Loading embeddings from JSON data...")
        embeddings_matrix = np.asarray(jsondic["embeddings"], dtype=np.float32)
      else:
        print('No embeddings in JSON data, please indicate word embedding size')
        exit()
    else:
      embeddings_matrix = None

    if FLAGS.use_binary_features:
      if jsondic.get("featdim", None):
        print("Found binary features, adding them.")
        train_feat_x = jsondic["train_feat_x"]
        valid_feat_x = jsondic["valid_feat_x"][:valid_limit]
        test_feat_x = jsondic["test_feat_x"][:test_limit]
        binary_feat_dim = jsondic["featdim"]
      else:
        print("No binary feats found in JSON data, please change flag an re run.")
        exit()
    else:
      train_feat_x = valid_feat_x = test_feat_x = None
      binary_feat_dim = 0

    vocab = jsondic["token2idx"]
    rev_vocab = {idx: token for token, idx in vocab.items()}

    train = [train_x, train_feat_x, train_y, train_z, train_ids]
    valid = [valid_x, valid_feat_x, valid_y, valid_z, valid_ids]
    test = [test_x, test_feat_x, test_y, test_z, test_ids]
    vocabs = [vocab, label_vocab]
    rev_vocabs = [rev_vocab, rev_label_vocab]

    return [train, valid, test, vocabs, rev_vocabs, binary_feat_dim, class_dim, embeddings_matrix]


def create_model(session, source_vocab_size, label_vocab_size,
                 word_embedding_matrix, binary_feat_dim, class_dim, FLAGS, _buckets):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = MultiTaskModel(source_vocab_size, label_vocab_size, _buckets,
                                 FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers,
                                 FLAGS.max_gradient_norm, FLAGS.batch_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 zoneout_keep_prob=FLAGS.zoneout_keep_prob,
                                 rnn=FLAGS.rnn,
                                 forward_only=False, use_attention=FLAGS.use_attention,
                                 bidirectional_rnn=FLAGS.bidirectional_rnn, task=FLAGS.task,
                                 context_win_size=FLAGS.context_win_size,
                                 word_embedding_matrix=word_embedding_matrix,
                                 learning_rate=FLAGS.learning_rate,
                                 decay_factor=FLAGS.decay_factor,
                                 decay_end=FLAGS.decay_end,
                                 use_binary_features=FLAGS.use_binary_features,
                                 binary_feat_dim=binary_feat_dim,
                                 class_dim=class_dim,
                                 optimizer=FLAGS.optimizer,
                                 loss=FLAGS.loss,
                                 train_embeddings=FLAGS.train_word_embeddings,
                                 regularization_lambda=FLAGS.regularization_lambda)

  with tf.variable_scope("model", reuse=True):
    model_test = MultiTaskModel(source_vocab_size, label_vocab_size, _buckets,
                                FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers,
                                FLAGS.max_gradient_norm, FLAGS.eval_batch_size,
                                dropout_keep_prob=FLAGS.dropout_keep_prob,
                                zoneout_keep_prob=FLAGS.zoneout_keep_prob,
                                rnn=FLAGS.rnn,
                                forward_only=True, use_attention=FLAGS.use_attention,
                                bidirectional_rnn=FLAGS.bidirectional_rnn, task=FLAGS.task,
                                context_win_size=FLAGS.context_win_size,
                                word_embedding_matrix=word_embedding_matrix,
                                learning_rate=FLAGS.learning_rate,
                                decay_factor=FLAGS.decay_factor,
                                decay_end=FLAGS.decay_end,
                                use_binary_features=FLAGS.use_binary_features,
                                binary_feat_dim=binary_feat_dim,
                                class_dim=class_dim,
                                optimizer=FLAGS.optimizer,
                                loss=FLAGS.loss,
                                train_embeddings=FLAGS.train_word_embeddings,
                                regularization_lambda=FLAGS.regularization_lambda)

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())

    if FLAGS.word_embedding_size == 0 and word_embedding_matrix is not None:
      with tf.variable_scope("model", reuse=True):
        embedding = tf.get_variable("embedding")
        session.run(embedding.assign(word_embedding_matrix))
  return model_train, model_test


def run_valid_test(model_test, sess, data_set, mode, task,
                   rev_vocab, rev_label_vocab, FLAGS, _buckets,
                   classification_out_file=None, regression_out_file=None,
                   attentions_out_file=None):
    # Run evals on development/test set and print the accuracy.
    ref_values_list, hyp_values_list = [], []
    ref_label_list, hyp_label_list = [], []
    word_idx_list = []
    class_attentions = []
    reg_attentions = []
    example_ids = []
    correct_count = 0
    accuracy = 0.0

    model_batch_size = model_test.batch_size
    for bucket_id in xrange(len(_buckets)):
        eval_loss = 0.0
        count = 0
        while count < len(data_set[bucket_id]):
            #if model_batch_size == 1:
            #  model_inputs = model_test.get_one(data_set, bucket_id, count)
            #else:
            model_inputs = model_test.get_batch(data_set, bucket_id, count)

            encoder_inputs, simple_encoder_inputs, encoder_extra_inputs, \
            sequence_lengths, values, labels, ids = model_inputs

            reg_values = []
            class_logits = []

            if task['joint'] == 1:
              output = model_test.joint_step(sess, encoder_inputs, encoder_extra_inputs,
                                             values, labels, sequence_lengths, bucket_id, True)
              _, step_loss, reg_values, reg_att, class_logits, class_att = output

            elif task['classification'] == 1:
              output = model_test.classification_step(sess, encoder_inputs, encoder_extra_inputs,
                                                      labels, sequence_lengths, bucket_id, True)
              _, step_loss, class_logits, class_att = output

            elif task['regression'] == 1:
                output = model_test.regression_step(sess, encoder_inputs, encoder_extra_inputs,
                                                    values, labels, sequence_lengths, bucket_id, True)
                _, step_loss, reg_values, reg_att = output

            eval_loss += step_loss / len(data_set[bucket_id])

            example_ids += ids

            for seq_id, sequence_length in enumerate(sequence_lengths):
              current_word_idx_list = []
              for token_id in xrange(sequence_length):
                current_word_idx_list.append(simple_encoder_inputs[token_id][seq_id])
              word_idx_list.append(current_word_idx_list)
              if task['regression'] == 1:
                reg_attentions.append(reg_att[seq_id][:sequence_length].tolist())
              if task['classification'] == 1:
                class_attentions.append(class_att[seq_id][:sequence_length].tolist())

            if task['regression'] == 1:
              for seq_id in range(len(sequence_lengths)):
                hyp_value = reg_values[seq_id]
                hyp_values_list.append(hyp_value)
                if mode != "Test":
                  ref_value = values[0][seq_id]
                  ref_values_list.append(ref_value)

            for seq_id in range(len(sequence_lengths)):
              ref_label = rev_label_vocab[labels[0][seq_id]]
              ref_label_list.append(ref_label)
              if task['classification'] == 1:
                hyp_class = np.argmax(class_logits[seq_id])
                hyp_label = rev_label_vocab[hyp_class]
                hyp_label_list.append(hyp_label)
                if ref_label == hyp_label:
                  correct_count += 1

            count += model_batch_size

    word_list = [[rev_vocab[idx] for idx in sequence] for sequence in word_idx_list]
    if mode != "Test":
      regression_mean_error = np.mean([abs(ref - hyp) for ref, hyp in zip(ref_values_list, hyp_values_list)])
    else:
      regression_mean_error = 0
    if FLAGS.task['regression'] == 1:
        print("  %s regression mean error: %.2f" % (mode, float(regression_mean_error)))
        regression_eval_result = regeval(word_list, hyp_values_list, ref_values_list, ref_label_list,
                                         example_ids, regression_out_file=regression_out_file)
        pear, spear, pear_05, spear_05 = regression_eval_result
        print("  %s regression mean pearson: %.2f  regression mean spearman: %.2f " % (mode, pear, spear))
        print("  %s regression top 0.5 mean pearson: %.2f  regression top 0.5 mean spearman: %.2f " % (mode, pear_05, spear_05))
        sys.stdout.flush()

    if FLAGS.task['classification'] == 1:
        accuracy = float(correct_count) * 100 / count
        print("  %s classification accuracy: %.2f %d/%d" % (mode, accuracy, correct_count, count))
        classification_eval_result = classeval(hyp_label_list, ref_label_list, rev_label_vocab.values(),
                                               classification_out_file)
        for label, cm in classification_eval_result.items():
          print("  \t%s accuracy: %.2f  precision: %.2f  recall: %.2f  f1-score: %.2f" % (
            label, 100 * cm["a"], 100 * cm["p"], 100 * cm["r"], 100 * cm["f1"]))
        sys.stdout.flush()

    if attentions_out_file:
      json_att_dic = {"regression_atts": reg_attentions,
                      "sequences": word_list}

      with open(attentions_out_file, "w") as f:
        json.dump(json_att_dic, f)

    return (regression_mean_error, pear, spear, pear_05, spear_05 ), accuracy

if __name__ == "__main__":
  main()
