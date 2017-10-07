# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import urllib.parse as urllib
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        # Des this make "fc2/add" the "output_node_names" for convert_variables_to_constants??
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # batch = mnist.train.next_batch(FLAGS.training_batch)
    # print(json.dumps(batch[0][0].tolist()))
    # return
    #

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            saver.restore(sess, FLAGS.session_dir)
            print('Restored session from: %s' % FLAGS.session_dir)
        except tf.OpError:
            sess.run(tf.global_variables_initializer())
        if FLAGS.classify is None and FLAGS.rest is None:
            graph_location = FLAGS.graph_dir

            # TensorBoard
            # print('Saving graph to: %s' % graph_location)
            # train_writer = tf.summary.FileWriter(graph_location)
            # train_writer.add_graph(tf.get_default_graph())

            mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
            for i in range(101):
                batch = mnist.train.next_batch(FLAGS.training_batch)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g  loss %g' % (i, train_accuracy, loss))
                if i % 500 == 0:
                    print('Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
                    save_path = saver.save(sess, FLAGS.session_dir)
                    print('Model saved in file: %s' % save_path)

            print('Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            save_path = saver.save(sess, FLAGS.session_dir)
            print('Model saved in file: %s' % save_path)

            print('Freezing model...')
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                'fc2/add'.split(",")  # The output node names are used to select the useful nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(FLAGS.session_dir + '-frozen.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))

            # print('---------------------')
            # print('Ops on default graph:')
            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)
        elif FLAGS.classify is not None:
            file = FLAGS.classify
            print('Classifying image %s' % file)
            img = Image.open(file)
            img = img.resize((28, 28))
            img = np.array(img) * 255
            pred = sess.run(y_conv, feed_dict={x: [img.flatten()], keep_prob: 1.0})
            print('PRED')
            print(pred)
            print(np.argmax(pred))
        elif FLAGS.rest is not None:
            print('Running...')
            host = "localhost"
            port = 6006

            class Server(BaseHTTPRequestHandler):
                def do_GET(self):
                    query = urllib.parse_qs(urllib.urlparse(self.path).query)
                    data = {}

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    try:
                        if 'file' in query:
                            file = '/Users/rsilveira/Desktop/' + str(query['file'][0])
                            data['file'] = file
                            img = Image.open(file)
                            img = img.resize((28, 28))
                            img = np.array(img) * 255
                        elif 'data' in query:
                            data['data'] = query.data
                            img = np.array(json.loads(query['data']))
                            img = img.resize((28, 28))

                        pred = sess.run(y_conv, feed_dict={x: [img.flatten()], keep_prob: 1.0})
                        data['predictions'] = pred.flatten().tolist()
                        data['prediction'] = int(np.argmax(pred))
                        data['classes'] = [n for n in range(10)]
                        print(data)
                        self.wfile.write(bytes(json.dumps(data), "utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.wfile.write(bytes(json.dumps({'error': str(e)}), "utf-8"))

                def do_POST(self):
                    data = {}

                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    post_data = json.loads(post_data)
                    pixels = list()

                    if 'pixels' in post_data:
                        pixels = post_data['pixels']
                        print('PIXELS: ' + str(len(post_data['pixels'])))

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()

                    if len(pixels) < 28 * 28:
                        self.send_response(400)
                        data['error'] = 'Invalid input. Pixels array must be ' + str(28 * 28) + ' elements.'
                    else:
                        try:
                            pred = sess.run(y_conv, feed_dict={x: [pixels], keep_prob: 1.0})
                            data['predictions'] = pred.flatten().tolist()
                            data['prediction'] = int(np.argmax(pred))
                            data['classes'] = [n for n in range(10)]
                            print(data)
                        except Exception as e:
                            self.send_response(500)
                            data['error'] = str(e)

                    self.wfile.write(bytes(json.dumps(data), "utf-8"))
                    # data = {}
                    #
                    # content_length = int(self.headers['Content-Length'])
                    # post_data = self.rfile.read(content_length)
                    # post_data = json.loads(post_data)
                    # pixels = list()
                    #
                    # if 'pixels' in post_data:
                    #     pixels = post_data['pixels']
                    #     print('PIXELS: ' + str(len(post_data['pixels'])))
                    #
                    # self.send_header("Content-type", "application/json")
                    # if len(pixels) < 28 * 28:
                    #     self.send_response(400)
                    #     data['error'] = 'Invalid input. Pixels array must be ' + str(28 * 28) + ' elements.'
                    # else:
                    #     try:
                    #         # pred = sess.run(y_conv, feed_dict={x: [pixels], keep_prob: 1.0})
                    #         # data['predictions'] = pred.flatten().tolist()
                    #         # data['prediction'] = int(np.argmax(pred))
                    #         data['classes'] = [n for n in range(10)]
                    #         self.send_response(200)
                    #     except Exception as e:
                    #         self.send_response(500)
                    #         data['error'] = str(e)
                    #
                    # print(data)
                    # self.end_headers()
                    # self.wfile.write(bytes(json.dumps(data), "utf-8"))

            server = HTTPServer((host, port), Server)
            print(time.asctime(), "Server Starts - %s:%s" % (host, port))

            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass

            server.server_close()
            print(time.asctime(), "Server Stops - %s:%s" % (host, port))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.getcwd() + '/data',
                        help='Directory for storing input data')
    parser.add_argument('--graph_dir', type=str,
                        default=os.getcwd() + '/graph',
                        help='Directory for storing graph data')
    parser.add_argument('--session_dir', type=str,
                        default=os.getcwd() + '/session',
                        help='Directory for storing session data')
    parser.add_argument('--training_batch', type=str,
                        default=128,
                        help='Size of training batch')
    parser.add_argument('--classify', type=str,
                        default=None,
                        help='Path of file to classify')
    parser.add_argument('--rest', type=str,
                        default=None,
                        help='Run script as daemon')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
