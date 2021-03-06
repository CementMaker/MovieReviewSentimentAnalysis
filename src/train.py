import os
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from Model import *
from sklearn.metrics import classification_report


def get_batch(epoches, batch_size):
    train_x, train_y = pickle.load(open("./pkl/train.pkl", "rb"))
    data = list(zip(train_x, train_y))

    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(train_y), batch_size):
            if batch + batch_size < len(train_y):
                yield data[batch: (batch + batch_size)]


def train_step(model, batch, label):
    feed_dict = {
        model.model.sentence: batch,
        model.model.label: label,
        model.model.dropout_keep_prob: 0.5
    }
    _, summary, step, loss, accuracy, = model.sess.run(
        fetches=[model.optimizer, model.merged_summary_train, model.global_step, model.model.loss, model.model.accuracy],
        feed_dict=feed_dict)
    model.summary_writer_train.add_summary(summary, step)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))


def dev_step(model, batch, label, return_predict=False):
    feed_dict = {
        model.model.sentence: batch,
        model.model.label: label,
        model.model.dropout_keep_prob: 1.0
    }
    summary, step, loss, accuracy, predict = model.sess.run(
        fetches=[model.merged_summary_test, model.global_step, model.model.loss, model.model.accuracy, model.model.predict],
        feed_dict=feed_dict)
    model.summary_writer_test.add_summary(summary, step)
    print("test: step {}, loss {}, accuracy {}".format(step, loss, accuracy))
    if return_predict == 1:
        return predict


class NeuralBowTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = NeuralBOW.NeuralBOW(sentence_length=50,
                                         embedding_size=150,
                                         vocab_size=18131,
                                         num_label=5)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(7, 400)

        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/nbow_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/nbow_summary/test", graph=self.sess.graph)


class textCnnTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = textCnn.Cnn(sequence_length=60,
                                 embedding_size=100,
                                 filter_sizes=[1, 2, 3, 4],
                                 num_filters=10,
                                 num_classes=5,
                                 vocab_size=20000)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 400)

        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/cnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/cnn_summary/test", graph=self.sess.graph)


class textRnnTrain(object):
    def __init__(self):
        self.sess = tf.Session()
        self.model = textLstm.lstm(num_layers=1,
                                   sequence_length=50,
                                   embedding_size=100,
                                   vocab_size=20000,
                                   rnn_size=100,
                                   num_classes=5)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(10, 400)

        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/rnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/rnn_summary/test", graph=self.sess.graph)


def main(model):
    test_x, test_y = pickle.load(open("./pkl/test.pkl", "rb"))
    for data in model.batches:
        x_train, y_train = zip(*data)
        train_step(model, x_train, y_train)
        current_step = tf.train.global_step(model.sess, model.global_step)
        if current_step % 10 == 0:
            print("\n:")
            dev_step(model, test_x, test_y)

    predict = dev_step(model, test_x, test_y, return_predict=True)
    print(classification_report(y_true=test_y, y_pred=predict))


if __name__ == "__main__":
    Net = NeuralBowTrain()
    main(Net)
