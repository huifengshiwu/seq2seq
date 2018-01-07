import tensorflow as tf


sess = tf.InteractiveSession()


x = tf.Variable(1, dtype=tf.int32, trainable=False)
b = tf.placeholder(dtype=tf.bool, shape=[])

def f():
    return x.assign(2)

def g():
    return x.assign(3)


y = tf.cond(
    b, f, g
)

sess.run(tf.global_variables_initializer())

#import ipdb; ipdb.set_trace()
print(sess.run([y, x], {b: True}))
print(sess.run([y, x], {b: False}))
print(sess.run([y, x], {b: True}))