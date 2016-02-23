#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))

py_x = tf.matmul(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in xrange(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end]})
    print i
    print np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X:teX, Y:teY}))
