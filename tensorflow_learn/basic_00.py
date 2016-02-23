#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

def example_00():
    a = tf.placeholder("float")
    b = tf.placeholder("float")

    y = tf.mul(a,b)

    sess = tf.Session()
    print sess.run(y,feed_dict={a:10,b:19})

def linear_regression():
    trX = np.linspace(-1,1,101)
    trY = 2 * trX + np.random.randn(trX.shape[0]) * 0.33

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    def model(X,w):
        return tf.mul(X,w)

    w = tf.Variable(0.0,name="weights")
    y_model = model(X,w)

    cost = tf.pow(Y-y_model,2)

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in xrange(100):
        for (x,y) in zip(trX,trY):
            sess.run(train_op,feed_dict={X:x,Y:y})
    print sess.run(w)


if __name__ == "__main__":
    linear_regression()

