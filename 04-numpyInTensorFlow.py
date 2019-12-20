import tensorflow as tf
import numpy as np

#start the session
sess = tf.Session()

zeroD = np.array(30, dtype=np.int32)
print(sess.run(tf.rank(zeroD)))
print(sess.run(tf.shape(zeroD)))

oneD = np.array([1.3, 4.3, 5.4, 43.2], dtype=np.float32)
print(sess.run(tf.rank(oneD)))
print(sess.run(tf.shape(oneD)))

sess.close()
