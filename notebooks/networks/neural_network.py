import tensorflow as tf
import numpy as np

w = 2
h = 3
c = 4
x = np.zeros((h, w)) + np.array([i for i in range(h * w)]).reshape((h, w))
# W = np.zeros((w, c)) + np.array([i for i in range(w * c)]).reshape((w, c))
# b = np.zeros((c, 1)) + np.array([i for i in range(c)]).reshape((c, 1))

_x = tf.placeholder(tf.float32, [None, w])
_y = tf.placeholder(tf.float32, [None, c])
_W = tf.Variable(tf.ones([w, c]))
_b = tf.Variable([[1., 2., 3., 4.]])

logits = tf.matmul(_x, _W) + _b
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    l = session.run(logits, feed_dict={_x: x})
    print(x)
    print('--')
    print(l)
