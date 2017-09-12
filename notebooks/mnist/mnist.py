import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch = mnist.train.next_batch(5)
print('mnist')
print(batch[0].shape)

plt.imshow(batch[0][0].reshape((28, 28)), cmap='gray')
plt.show()
