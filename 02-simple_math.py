import tensorflow as tf

# 1D tensor
x = tf.constant([100, 200, 300], name='x')

# another 1D tensor
y = tf.constant([1, 2, 3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')
final_div = tf.divide(sum_x, prod_y, name='final_div')
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

session = tf.Session()
print('x : ', session.run(x))
print('y: ', session.run(y))
print('sum x: ', session.run(sum_x))
print('prod y: ', session.run(prod_y))
print('sum(x)/prod(y): ', session.run(final_div))
print('mean(sum(x), prod(y)) : ', session.run(final_mean))

# output to tensorboard
writer = tf.summary.FileWriter('./output', session.graph)

writer.close()
session.close()

# then, I need to write the following code in the terminal
# tensorboard --logdir="output"



