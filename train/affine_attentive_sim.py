# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:21:21 2017

@author: samsung
"""
import tensorflow as tf
import numpy as np

#from tensorflow.contrib.rnn import stack_bidirectional_rnn as bi_rnn


class Affine(object):
# Combine all the pooled features
    def __init__(
      self, sequence_length_head, sequence_length_body, num_classes, vocab_size_head, vocab_size_body,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.1):
        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_x_head = tf.placeholder(tf.int32, [None, sequence_length_head], name="input_x_head")
        self.input_x_body = tf.placeholder(tf.int32, [None, sequence_length_body], name="input_x_body")
        
        # Embedding layer
        self.embeddings_head = tf.Variable(
                tf.random_uniform([vocab_size_head, embedding_size], -1.0, 1.0),trainable=False)#trainable=false
        self.embedded_chars_head = tf.nn.embedding_lookup(self.embeddings_head, self.input_x_head)
        self.embedded_chars_expanded_head = tf.expand_dims(self.embedded_chars_head, -1)

        self.embeddings_body = tf.Variable(
                tf.random_uniform([vocab_size_body, embedding_size], -1.0, 1.0),trainable=False)#trainable=false
        self.embedded_chars_body = tf.nn.embedding_lookup(self.embeddings_body, self.input_x_body)
        self.embedded_chars_expanded_body = tf.expand_dims(self.embedded_chars_body, -1)
        
        self.pooled_outputs_head = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-head-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, 256]
                W_head = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_head")
                b_head = tf.Variable(tf.constant(0.1, shape=[256]), name="b_head")
                conv_head = tf.nn.conv2d(
                    self.embedded_chars_expanded_head,
                    W_head,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                self.h_head = tf.nn.relu(tf.nn.bias_add(conv_head, b_head), name="relu_head")

        self.pooled_outputs_body = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-body-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, 1024]
                W_body = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_body")
                b_body = tf.Variable(tf.constant(0.1, shape=[1024]), name="b_body")
                conv_body = tf.nn.conv2d(
                    self.embedded_chars_expanded_body,
                    W_body,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                self.h_body = tf.nn.relu(tf.nn.bias_add(conv_body, b_body), name="relu_body")
                # Maxpooling over the outputs
        
        l2_loss = tf.constant(0.0)
        self.num_filters_total = num_filters * len(filter_sizes)
        self.U = tf.Variable(tf.truncated_normal(shape = [256,1024],stddev = 0.01,name = 'U'))
        self.pooled_outputs_head, self.pooled_outputs_body = self.attentive_pooling(self.h_head,self.h_body,sequence_length_head,sequence_length_body)
        self.sims = self.interact(self.pooled_outputs_head, self.pooled_outputs_body)
        pooled_outputs = tf.concat([self.pooled_outputs_head,self.sims,self.pooled_outputs_body],-1,name='preconcat')
        
        self.h_pool = tf.concat(pooled_outputs, 3, name='concat')
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total+1])
       
       	
        W_fc1 = tf.Variable(tf.truncated_normal([1281,1024],stddev=0.1),name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]),name="b_fc1")
        h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat,W_fc1) + b_fc1)
            
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.W = tf.get_variable(
                "W",
                shape=[1024, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name="scores")
            self.probabilities = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            print(self.scores.shape)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            print("%d/%d",self.predictions,self.input_y)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attentive_pooling(self,input_left,input_right,sequence_length_head,sequence_length_body):
        
        head = tf.reshape(input_left,[-1,sequence_length_head-3+1,256],name = 'Q')
        body = tf.reshape(input_right,[-1,sequence_length_body-3+1,1024],name = 'A')
        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        print("head",head.shape)
        print("U",self.U.shape)
        first = tf.matmul(tf.reshape(head,[-1,256]),self.U)
        print("first",first.shape)
        second_step = tf.reshape(first,[-1,sequence_length_head - 3 + 1,1024])
        print("second_step",second_step.shape)
        result = tf.matmul(second_step,tf.transpose(body,perm = [0,2,1]))
        print("resultshape",result.shape)
        # print 'result',result
        G = tf.tanh(result)
        
        # G = result
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')
    
        self.attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        print(self.attention_q)
        self.see = self.attention_q

        self.attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')
        print(self.attention_a)
        R_q = tf.reshape(tf.matmul(head,self.attention_q,transpose_a = 1),[-1,256],name = 'R_q')
        R_a = tf.reshape(tf.matmul(self.attention_a,body),[-1,1024],name = 'R_a')
        print(R_q)
        print(R_a)

        return R_q,R_a
    
    def interact(self, head, body):
        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W_sim",
                shape=[256, 1024],
                initializer=tf.contrib.layers.xavier_initializer())
            # print 'q_pooling',self.q_pooling
            # print 'num_filters',self.num_filters_total
            self.transform_head = tf.matmul(head, W)
            print(self.transform_head)
            sims = tf.reduce_sum(tf.multiply(self.transform_head, self.body), 1, keep_dims=True)
            
        return sims