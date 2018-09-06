import tensorflow as tf

v1=tf.Variable(tf.constant(10.0),name="v1")
v2=tf.Variable(tf.constant(20.0),name="v2")


add_op=v1+v2
minus_op=v1-v2


with tf.control_dependencies([add_op,minus_op]):
    combine_op=tf.no_op("noop")      #便于后面同时调用加法和减法
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(add_op),sess.run(minus_op))