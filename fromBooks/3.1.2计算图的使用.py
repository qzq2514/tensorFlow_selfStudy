import tensorflow as tf

g1=tf.Graph()
with g1.as_default():
    # v=tf.Variable(tf.constant([2.0,3.0]),name="v")
    v=tf.get_variable("v",initializer=tf.constant([2.0,3.0]))


g2=tf.Graph()
with g2.as_default():
    v=tf.get_variable("v",initializer=tf.constant([22,33],dtype=tf.float32))



with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))



with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))           #不同的计算图有自己的变量，不会发生冲突




