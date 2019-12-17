import tensorflow as tf

# y = Ax^ + Bx + c
# Ax^2 + Bx^2

A = tf.constant([4], dtype=tf.int32, name='A')
B = tf.constant([5], dtype=tf.int32, name='B')
C = tf.constant([6], dtype=tf.int32, name='C')

x = tf.placeholder(dtype=tf.int32, name='x')

# y = Ax^ + Bx + c
with tf.name_scope("Equation_1"):
    Ax2 = tf.multiply(A, tf.pow(x,2), name='Ax2')
    Bx = tf.multiply(B, x, name='Bx')
    y1 = tf.add_n([Ax2, B, C], name='y1')

# Ax^2 + Bx^2
with tf.name_scope('Equation_2'):
    Ax2 = tf.multiply(A, pow(x, 2), name='Ax2')
    Bx2 = tf.multiply(B, tf.pow(B, 2), name='Bx2')
    y2 = tf.add_n([Ax2, Bx2], name='y2')

# the first two scopes feed to this scope!
with tf.name_scope('Final_sum'):
    y = y1 + y2

with tf.Session() as sess:
    print(sess.run(fetches=y, feed_dict={x:[10]}))

    writer = tf.summary.FileWriter('./m3_example6', sess.graph)
    writer.close()




