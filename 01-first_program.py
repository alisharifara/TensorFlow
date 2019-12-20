import tensorflow as tf

a = tf.constant(6.5, name='constant_a')
b = tf.constant(3.4, name='constant_b')
c = tf.constant(3.0, name='constant_c')
d = tf.constant(100.2, name='constant_d')

square = tf.square(a, name='square_a')
power = tf.pow(b, c, name='pow_b_c')
sqrt = tf.sqrt(d, name='sqrt_d')

final_sum = tf.add_n([square, power, sqrt], name='final_sum')

# start a new session
sess = tf.Session()

print('squre of a:', sess.run(square))
print('power of b and c:', sess.run(power))
print('sqrt of d:', sess.run(sqrt))
print('final sum:', sess.run(final_sum))

another_sum = tf.add_n([a, b, c, d, power], name='another_sum')

# to create an output to tensorboard
writer = tf.summary.FileWriter('./output', sess.graph)
writer.close()
sess.close()

# after that, you need to go to terminal and type the following:
# tensorboard --logdir="output"
