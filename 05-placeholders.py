import tensorflow as tf

# 1D vector with 3 values in it .. we name it x
x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='sum_y')

final_div = tf.div(sum_x, prod_y, name='final_div')
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sess = tf.Session()

print('sum(x)', sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print('prod(y)', sess.run(prod_y, feed_dict={y: [1, 2, 3]}))

# this needs to have both x and y placeholders for its calculation
print('sum(x)/prod(y)', sess.run(final_div, feed_dict={x: [10, 20, 30], y: [1, 2, 3]}))

# this needs to have both x and y placeholders for its calculation
print('mean (sum(x), prod(y))', sess.run(final_mean, feed_dict={x: [1000, 2000, 3000], y: [10, 20, 30]}))

writer = tf.summary.FileWriter('./m3_example1', sess.graph)

writer.close()
sess.close()

# then, in the terminal we can enter: tensorboard --logdir="m3_example1"
