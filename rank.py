import tensorflow as tf

session = tf.Session()

zeroD = tf.constant(4)
oneD = tf.constant(['ali', 'sharifara'])
twoD = tf.constant([[1.0, 2.9], [3.7, 4.4]])
threeD = tf.constant([[[1.0, 2.9], [3.7, 2.4]], [[3.7, 2.4], [3.7, 2.4]]])

print(session.run(tf.rank(zeroD)))
print(session.run(tf.rank(oneD)))
print(session.run(tf.rank(twoD)))
print(session.run(tf.rank(threeD)))

session.close()

