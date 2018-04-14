# -*- coding: utf-8 -*-:
"""
Created on Wed Nov  8 23:58:05 2017

@author: samsung
"""

import numpy as np
import codecs
import re
import itertools
from collections import Counter
from konlpy.tag import Komoran
from csv import DictReader
from csv import DictWriter

pos_tagger = Komoran()

class Data:
    def __init__(self, file_instances):
        # Load data
        self.instances = self.read(file_instances)
        self.headlines = {}
        self.bodies = {}
        self.labels = {}
         
        for instance in self.instances:
            instance['seqid'] = int(instance['\ufeffseqid'])
            
        for head in self.instances:
            self.headlines[head['seqid']] = head['title']
        
        for body in self.instances:
            self.bodies[body['seqid']] = body['content']
        
        for label in self.instances:
            self.labels[label['seqid']] = label['Label']
                
    def read(self, filename):
        
        rows = []
        # Process file
        with open(filename, "r") as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)
        return rows
    
    def get_data(self):
        return self.instances
        
def tokenize(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    final_string = ''
    tokenized = [''.join(t) for t in pos_tagger.pos(string) if t[1] in ['NNG','NNP','NNB','NR','VV','VA','VCP','VCN','XSV','XSA','SN','MAG','MM','MAJ']]
#not in ['ETM','EC','ETN','EF','EP','NF','NV','NA','SW','SO','SP','SF','SE','SS','IC','XSN','XPN']]
    for tokens in tokenized:
        final_string += tokens + " "
    return final_string

def flat(content):
    return ["{}{}".format(word, tag) for word, tag in pos_tagger.pos(content)]


def load_data_and_labels(file_instances):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    train_heads = []
    train_bodies = []
    train_labels = []
    data = Data(file_instances)
    
    for instance in data.instances:
        news_id = instance['seqid']
        train_label = instance['Label']
        train_heads.append(tokenize(data.headlines[news_id]))
        train_bodies.append(tokenize(data.bodies[news_id]))
        train_labels.append((news_id,train_label))

    results = np.zeros((len(train_labels),2))

    for i, train_label in train_labels:
        if train_label == '1':
            results[i,1] = 1
        else:
            results[i,0] = 1

    return train_heads, train_bodies, results


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_word_embedding(file_name, vocab_processor, embedding_dim):

    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), embedding_dim))
    print("Load word2vec file {}\n".format(file_name))
    with open(file_name, "rb") as f:
        for idx, line in enumerate(f):
            word = []
            vectors = []
            if idx == 0:
                vocab_size, dim = line.strip().split()
            else:
                tks = line.strip().split()
                word = tks[0].strip().decode('utf8')
                vectors = tks[1:]
                idx = vocab_processor.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = np.array(vectors)
    print(initW)
    return initW
