import tensorflow as tf
import numpy as np

# import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data

# store the MNIST dataset in tmp/data
# every image here is 28x28 image size = 784 pixels
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

# we only select 6000 images from the training batch
training_digits, training_labels = mnist.train.next_batch(6000)

# we only select 300 images from the test batch
test_digits, test_labels = mnist.test.next_batch(300)

# 784 is the size of each image -- None is the index of each image
training_digit_pl = tf.placeholder('float', [None, 784])

test_digit_pl = tf.placeholder('float', [784])

# nearest neighbor calculation using L1 distance
l1_distance = tf.abs(tf.add(training_digit_pl, tf.negative(test_digit_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: get min distance index (the nearest neighbor)
pred = tf.argmin(distance, 0)

accuracy = 0

# initializing the variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_digits)):

        nn_index = sess.run(pred,
                            feed_dict={training_digit_pl: training_digits,
                                       test_digit_pl: test_digits[i, :]})

        # argmax is the value of the specific digit
        print('test', i, 'prediction',
              np.argmax(training_labels[nn_index]),
              'true label', np.argmax(test_labels[i]))

        # Calculate the accuracy of KNN
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1/len(test_digits)

print('Accuracy:', accuracy)
