#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_droup_input, p_Drop_hidden):
    X = tf.nn.dropout(X, p_droup_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_Drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_Drop_hidden)

    return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
print trX.shape, trY.shape
print teX.shape, teY.shape

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in xrange(5):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X:trX[start: end], Y: trY[start: end], p_keep_input: 0.8, p_keep_hidden: 0.5})
    print i
    print np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0}))
