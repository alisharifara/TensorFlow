import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import os

filename = './pishi.jpg'

image = plt.imread(filename)

print('image shape:', image.shape)
print('image array:', image)

plt.imshow(image)
plt.show()


x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # original axis 0, 1, 2
    # transpose = tf.transpose(x, perm=[1, 0, 2]) OR
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)
    # height and width of the image swapped
    print('shape of the transposed image:', result.shape)
    plt.imshow(result)
    plt.show()








