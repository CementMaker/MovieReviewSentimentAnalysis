import os
import sys
import pickle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.contrib import learn
from tensorflow import data


class feature(object):
    def __init__(self):
        self.train_df = pd.read_csv("../data/train.tsv", sep='\t')
        self.train_sentences = self.train_df['Phrase'].values
        self.train_Sentiment = self.train_df['Sentiment'].values
        self.one_hot_label = np.zeros(shape=(self.train_Sentiment.shape[0], 5))
        self.one_hot_label[np.arange(0, self.train_Sentiment.shape[0]), self.train_Sentiment] = 1

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(50, min_frequency=0)
        self.all_context = np.array(list(self.vocab_processor.fit_transform(self.train_sentences)))

        print("number of words :", len(self.vocab_processor.vocabulary_))
        print(self.all_context.shape, self.one_hot_label.shape)

        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.all_context, self.one_hot_label, test_size=0.02)
        pickle.dump((self.train_x, self.train_y), open("./pkl/train.pkl", "wb"))
        pickle.dump((self.test_x, self.test_y), open("./pkl/test.pkl", "wb"))


class ProcessFastText(object):
    def __init__(self):
        self.test_df = pd.read_csv("../data/test.tsv", sep='\t')
        self.train_df = pd.read_csv("../data/train.tsv", sep='\t')

        self.test_sentences = self.test_df['Phrase'].values
        self.train_sentences = self.train_df['Phrase'].values
        self.train_Sentiment = self.train_df['Sentiment'].values

        fd_train = open('../data/trainOnline.txt', 'w+')
        for label, sentence in zip(self.train_Sentiment, self.train_sentences):
            fd_train.write("__label__" + str(label) + '\t' + sentence + '\n')
        fd_train.close()

        fd_test = open('../data/testOnline.txt', 'w+')
        for sentence in self.test_sentences:
            fd_test.write(sentence + '\n')
        fd_test.close()


        # self.train_x, self.test_x, self.train_y, self.test_y = \
        #     train_test_split(self.train_sentences, self.train_Sentiment, test_size=0.05)
        #
        # fd_train = open('../data/trainFastText.txt', 'w+')
        # for label, sentence in zip(self.train_y, self.train_x):
        #     fd_train.write("__label__" + str(label) + ', ' + sentence + '\n')
        # fd_train.close()
        #
        # fd_valid = open('../data/validFastText.txt', 'w+')
        # for label, sentence in zip(self.test_y, self.test_x):
        #     fd_valid.write(str(label) + '\t' + sentence + '\n')
        # fd_valid.close()


if __name__ == '__main__':
    f = ProcessFastText()
