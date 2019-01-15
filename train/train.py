# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:58:50 2017

@author: samsung
"""

#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from affine import Affine
from tensorflow.contrib import learn
import codecs
import operator

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 1280, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size",127, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

file_train_instances = "res.csv"

# Load data
print("Loading data...")
x_heads, x_bodies, y = data_helpers.load_data_and_labels(file_train_instances)

#bow_head = TfidfVectorizer(tokenizer=None, lowercase=False, max_features=128)
#bow_body = TfidfVectorizer(tokenizer=None, lowercase=False, max_features=1280)

#bow_head_vec = bow_head.fit_transform(x_heads)
#bow_body_vec = bow_body.fit_transform(x_bodies)

#bow_head_vocab = [v[0] for v in sorted(bow_head.vocabulary_.items(),key=operator.itemgetter(1))]
#print(x_heads)
#print(bow_head_vocab)

# Build vocabulary_head
max_document_length_head = max([len(x.split(" ")) for x in x_heads])
vocab_processor_head = learn.preprocessing.VocabularyProcessor(max_document_length=128)#bow_head max_features
x_head = np.array(list(vocab_processor_head.fit_transform(x_heads)))
# Build vocabulary_body
max_document_length_body = max([len(x.split(" ")) for x in x_bodies])
vocab_processor_body = learn.preprocessing.VocabularyProcessor(max_document_length=1280)
x_body = np.array(list(vocab_processor_body.fit_transform(x_bodies)))

print('----headline_shape----')
print(x_head.shape)
print('----body_shape----')
print(x_body.shape)
print('----label_shape----')
print(y.shape)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_head_shuffled = x_head[shuffle_indices]
x_body_shuffled = x_body[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train_head, x_dev_head = x_head_shuffled[:-6000], x_head_shuffled[-6000:]
x_train_body, x_dev_body = x_body_shuffled[:-6000], x_body_shuffled[-6000:]
y_train, y_dev = y_shuffled[:-6000], y_shuffled[-6000:]
print("Vocabulary Size_head: {:d}".format(len(vocab_processor_head.vocabulary_)))
print("Vocabulary Size_body: {:d}".format(len(vocab_processor_body.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print(x_train_head)
print(x_train_body)
print(x_train_head.shape)
print(x_train_body.shape)
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Affine(
            sequence_length_head=128,
            sequence_length_body=1280,
            num_classes=2,
            vocab_size_head=len(vocab_processor_head.vocabulary_),
            vocab_size_body=len(vocab_processor_body.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=500)

        # Write vocabulary
        vocab_processor_head.save(os.path.join(out_dir, "vocab_head"))
        vocab_processor_body.save(os.path.join(out_dir, "vocab_body"))
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        head_embedding = data_helpers.load_word_embedding("fasttext_3_10.vec",vocab_processor_head,FLAGS.embedding_dim)
        body_embedding = data_helpers.load_word_embedding("fasttext_3_10.vec",vocab_processor_body,FLAGS.embedding_dim)
        sess.run(cnn.embeddings_head.assign(head_embedding))
        sess.run(cnn.embeddings_body.assign(body_embedding))
 #       sess.run(cnn.cnn_head.W.assign(initW))
 #       sess.run(cnn.cnn_body.W.assign(initW))
            
        def train_step(x_batch_head, x_batch_body, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_head: x_batch_head,
              cnn.input_x_body: x_batch_body,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy, predictions = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print(predictions)
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch_head, x_batch_body, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x_head: x_batch_head,
              cnn.input_x_body: x_batch_body,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy, predictions = sess.run(
                [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            return accuracy, loss, predictions
            #if writer:
            #    writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train_head, x_train_body, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
      
        # Training loop. For each batch...
        for batch in batches:
            x_batch_head, x_batch_body, y_batch = zip(*batch)
            train_step(x_batch_head, x_batch_body, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                batches_dev = data_helpers.batch_iter(list(zip(x_dev_head, x_dev_body, y_dev)),31, 1)
                acc_total, loss_total, total = 0,0,0
                for batch_dev in batches_dev:
                    x_batch_dev_head, x_batch_dev_body, y_batch_dev = zip(*batch_dev)
                    acc_dev, loss_dev, predictions = dev_step(x_batch_dev_head, x_batch_dev_body, y_batch_dev)
                    acc_total = acc_total + acc_dev
                    loss_total = loss_total + loss_dev
                    print(predictions)
                    total = total + 1
                results = codecs.open('results_all_new.txt','a')
                result = "total loss {:g}, total acc {:g}".format(loss_total/total, acc_total/total)
                print(result)
                results.write("step{:g}, total loss{:g}, total acc{:g}".format(current_step, loss_total/total, acc_total/total) + '\n')
                results.close()
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print(path)
                print("Saved model checkpoint to {}\n".format(path))
