
import os
import cv2
import glob
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import tensorflow as tf

# Loading train data
PIXELS = 24
imageSize = PIXELS * PIXELS
num_features = imageSize

def load_train_cv(encoder):
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = cv2.imread(fl,0)
            img = cv2.resize(img, (PIXELS, PIXELS))
            #img = img.transpose(2, 0, 1)
            img = np.reshape(img, (num_features))
            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = encoder.fit_transform(y_train).astype('int32')

    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    X_train = X_train/  255.
    X_test = X_test / 255.

    return X_train, y_train, X_test, y_test, encoder



def load_test():
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl,0)
        img = cv2.resize(img, (PIXELS, PIXELS))
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (num_features))
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    X_test = X_test / 255.

    return X_test, X_test_id



# load the training and validation data sets
encoder = LabelEncoder()
train_X, train_y, valid_X, valid_y, encoder = load_train_cv(encoder)
print('Train shape:', train_X.shape, 'Valid shape:', valid_X.shape)

# load data
#X_test, X_test_id = load_test()



# SETTINGs
LEARNING_RATE = 1e-4
TRAINING_ITERATION = 2500

DROPOUT = 0.5
BATCH_SIZE = 50









# Network Structure

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1,name='weight')
    return tf.Variable(initial)



def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape,name='bias')
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# images
x = tf.placeholder('float',shape=[None,imageSize])
# labels
y_ = tf.placeholder('int64',shape=[None])



# First layer  Convolution Layer

with tf.variable_scope('conv1'):
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

# (batch_size,571) => (batch_size,24,24,1)
image = tf.reshape(x,[-1,PIXELS,PIXELS,1])

# operation of convolution and max pooling
h_conv1 = tf.nn.relu(conv2d(image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# (batch_size,24,24,1) =>  (batch_size,12,12,32)


# Second Layer  Convolution Layer
with tf.variable_scope('conv2'):
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

# operation of convolution and max pooling
h_conv2 =  tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 =  max_pool_2x2(h_conv2)
# (batch_size,12,12,32) =>  (batch_size,6,6,64)



# Third Layer Fully Connecting
with tf.variable_scope('fc1'):
    W_fc1 = weight_variable([6 * 6 * 64,1024])
    b_fc1 = bias_variable([1024])

# (batch_size,6,6,64) => (batch_size,6 * 6 * 64)

h_pool2_flat =  tf.reshape(h_pool2,[-1,6 * 6 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# Dropout Layer prevent overfitting
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax regression
with tf.variable_scope('fc2'):
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

y = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
# Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)





# Stochastic gradient descent
epochs_completed = 0
index_in_epoch = 0
num_examples = train_X.shape[0]

def next_batch(batch_size):

    global train_X
    global train_y
    global index_in_epoch
    global epochs_completed
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_X = train_X[perm]
        train_y = train_y[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return train_X[start:end],train_y[start:end]




# initial the network
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)



train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1


for i in range(TRAINING_ITERATION):
    # get new batch
    batch_xs,batch_ys =  next_batch(BATCH_SIZE)

    # train on batch
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:DROPOUT})
    # check progress
    if i % display_step == 0 or (i + 1) ==TRAINING_ITERATION:
        train_cross_entropy = cross_entropy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
        validation_cross_entropy = cross_entropy.eval(feed_dict={x:valid_X,y_:valid_y,keep_prob:1.0})
        print 'train_cross_entropy => %.2f validation cross entropy => %.2f for step %d' % (train_cross_entropy,validation_cross_entropy,i)








