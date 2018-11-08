#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:29:05 2018

@author: wuxian
"""

from flask import Flask, request
import tensorflow as tf
from scipy import misc
import logging
import logging.config
from werkzeug.utils import secure_filename
import sys
import cass


app = Flask(__name__)


tf.reset_default_graph() 

x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver.restore(sess, 'model/model.ckpt') 
prediction=tf.argmax(y_conv,1)

def pred_int(input):
    return sess.run(prediction, feed_dict={x: input, keep_prob: 1.0})
    

@app.route("/upload", methods=['POST'])
def upload():
    f = request.files['file']
    im = misc.imread(f)
    fname = secure_filename(f.filename)
    img = im.reshape((1,784))

    
    output = pred_int(img)
    
    
    app.logger.info("%s, %s", fname, output[0])
    cass.insertData()
    
    return 'result: %s ' % (output[0])
    
    
    
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''  

if __name__ == "__main__":
    app.debug = True
    handler = logging.FileHandler('flask.txt')
    logging_format = logging.Formatter(
        '%(asctime)s, %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)  
    
                              
    app.run(port=3000, host='0.0.0.0')
   
