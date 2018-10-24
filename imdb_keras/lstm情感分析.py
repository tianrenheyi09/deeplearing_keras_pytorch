# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:13:33 2018

@author: Administrator
"""

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk  #用来分词
import collections  #用来统计词频
import numpy as np

maxlen = 0  #句子最大长度
word_freqs = collections.Counter()  #词频
num_recs = 0 # 样本数
with open('./data/train_data.txt','r+',encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./data/train_data.txt','r+',encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
