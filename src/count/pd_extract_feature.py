import os
import sys
import pickle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class feature(object):
    def __init__(self):
        self.test_df = pd.read_csv("../data/test.tsv", sep='\t')
        self.train_df = pd.read_csv("../data/train.tsv", sep='\t')

        self.train_sentences = self.train_df['Phrase'].values
        self.train_Sentiment = self.train_df['Sentiment'].values

        self.test_PhraseId = self.test_df['PhraseId'].values
        self.test_sentences = self.test_df['Phrase'].values

    # 统计句子长度
    def count_sentence_length(self):
        train_length, test_length = [], []
        for sentence in self.train_sentences: train_length.append(len(sentence.strip().split()))
        for sentence in self.train_sentences: test_length.append(len(sentence.strip().split()))

        train_length = sorted(Counter(train_length).items(), key=lambda x: x[0], reverse=False)
        test_length = sorted(Counter(test_length).items(), key=lambda x: x[0], reverse=False)
        plt.plot([a for a, b in train_length], [b for a, b in train_length])
        plt.plot([a for a, b in test_length], [b for a, b in test_length])
        plt.savefig('../picture/sentence_length.png')

    def feature_tf_idf(self):
        for index in range(len(self.train_sentences)):
            self.train_sentences[index] = self.train_sentences[index].strip('\n')
        for index in range(len(self.test_sentences)):
            self.test_sentences[index] = self.test_sentences[index].strip('\n')

        self.vectorizer = CountVectorizer(max_df=0.5,
                                          max_features=5000,
                                          min_df=2,
                                          lowercase=False,
                                          decode_error='ignore',
                                          analyzer=str.split).fit(self.train_sentences)
        self.test_sentences = self.vectorizer.transform(self.test_sentences)
        self.train_sentences = self.vectorizer.transform(self.train_sentences)

        self.train_context, self.test_context, self.train_label, self.test_label = train_test_split(self.train_sentences,
                                                                                                    self.train_Sentiment,
                                                                                                    test_size=0.2)
        pickle.dump((self.train_sentences, self.train_Sentiment), open('../data/pickle/data.pickle', "wb"))
        pickle.dump((self.test_sentences, self.test_PhraseId), open('../data/pickle/test_data.pickle', "wb"))
        pickle.dump((self.train_context, self.train_label), open('../data/pickle/train.pickle', "wb"))
        pickle.dump((self.test_context, self.test_label), open('../data/pickle/test.pickle', "wb"))


if __name__ == '__main__':
    feature().feature_tf_idf()
