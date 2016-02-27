#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division

import sys,os
import pickle,random
import numpy
import tensorflow as tf

NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64 
EVAL_BATCH_SIZE = BATCH_SIZE
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
N_OUTPUT_LAYER = 40
IMAGE_WIDTH, IMAGE_HEIGHT = 40,40

def error_rate(predictions, labels):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])

def extract_data(filename):
    if os.path.exists(filename) == False:
        raise ValueError
        return None
    with open(filename,'rb') as handle:
        data = pickle.load(handle)
        return data

def fetchDataSet(img_path,labels_path):
    img_data = extract_data(img_path)
    img_label = extract_data(labels_path) - 1
    ind_list = range(0, len(img_label))
    random.shuffle(ind_list)
    random.shuffle(ind_list)
    return img_data[ind_list],img_label[ind_list]

def getPaperData():
    # data info file path
    train_data_filename = "./ProcData/Paper/paper_train_data.pkl"
    train_labels_filename = "./ProcData/Paper/paper_train_label.pkl"
    train_data, train_labels = fetchDataSet(train_data_filename, train_labels_filename)
    total_len = len(train_labels)

    valid_data_filename = "./ProcData/Paper/paper_valid_data.pkl"
    valid_labels_filename = "./ProcData/Paper/paper_valid_label.pkl"
    valid_data, valid_labels = fetchDataSet(valid_data_filename, valid_labels_filename)

    test_data_filename = "./ProcData/Paper/paper_test_data.pkl"
    test_labels_filename = "./ProcData/Paper/paper_test_label.pkl"
    test_data, test_labels = fetchDataSet(test_data_filename, test_labels_filename)
    return train_data[:total_len // 2, ...], train_labels[:total_len // 2], test_data, test_labels, valid_data, valid_labels

def getDUCData():
    # data info file path
    train_data_filename = "./ProcData/DUC/DUC_train_data.pkl"
    train_labels_filename = "./ProcData/DUC/DUC_train_label.pkl"
    train_data, train_labels = fetchDataSet(train_data_filename, train_labels_filename)

    valid_data_filename = "./ProcData/DUC/DUC_valid_data.pkl"
    valid_labels_filename = "./ProcData/DUC/DUC_valid_label.pkl"
    valid_data, valid_labels = fetchDataSet(valid_data_filename, valid_labels_filename)

    test_data_filename = "./ProcData/DUC/DUC_test_data.pkl"
    test_labels_filename = "./ProcData/DUC/DUC_test_label.pkl"
    test_data, test_labels = fetchDataSet(test_data_filename, test_labels_filename)
    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels


def main():
    train_data, train_labels, test_data, test_labels, valid_data, valid_labels = getPaperData()
    train_size = train_labels.shape[0]
  
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE, ))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
    out_1, out_2 = 32, 64
    temp_size = 3
    conv1_weights = tf.Variable(tf.truncated_normal([temp_size, temp_size, NUM_CHANNELS, out_1], stddev=0.1,seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([out_1]))
    conv2_weights = tf.Variable(tf.truncated_normal([temp_size, temp_size, out_1, out_2], stddev=0.1, seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[out_2]))

    fc23_weights = tf.Variable(tf.truncated_normal([IMAGE_WIDTH // 4 * IMAGE_HEIGHT // 4 * out_2, 128], stddev=0.1, seed=SEED))
    fc23_biases = tf.Variable(tf.constant(0.1, shape=[128]))
    fc3o_weights = tf.Variable(tf.truncated_normal([128, N_OUTPUT_LAYER], stddev=0.1, seed=SEED))
    fc3o_biases = tf.Variable(tf.constant(0.1, shape=[N_OUTPUT_LAYER]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        ############
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        ##########
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        print "Pool Shape ", pool_shape
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        print "Reshape ", reshape.get_shape().as_list()
        print "fc1_weights ", fc23_weights.get_shape().as_list()
        print 
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc23_weights) + fc23_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc3o_weights) + fc3o_biases


  # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc23_weights) + tf.nn.l2_loss(fc23_biases) + tf.nn.l2_loss(fc3o_weights) + tf.nn.l2_loss(fc3o_biases))
  # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
    batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
  # Use simple momentum for the optimization.
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
    eval_data = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
    eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        predictions = numpy.ndarray(shape=(size, N_OUTPUT_LAYER), dtype=numpy.float32)
        for begin in xrange(0, size, BATCH_SIZE):
            end = begin + BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions


  # Create a local session to run the training.
    sess = tf.Session()
    # Run all the initializers to prepare the trainable parameters.
    init = tf.initialize_all_variables()
    sess.run(init)
    print('Initialized!')
    print "train size is", train_size
    # Loop through training steps.
    for i in xrange(100):
        for start, end in zip(range(0,train_size,BATCH_SIZE),range(BATCH_SIZE,train_size,BATCH_SIZE)):
          # Compute the offset of the current minibatch in the data.
          # Note that we could use better randomization across epochs.
            batch_data = train_data[start:end, ...]
            batch_labels = train_labels[start:end]
          # This dictionary maps the batch data (as a numpy array) to the
          # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
          # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
            if start % EVAL_FREQUENCY == 0:
                print "->" * 20
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(eval_in_batches(valid_data,sess),valid_labels))
                sys.stdout.flush()

    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    sess.close()

if __name__ == '__main__':
    main()
