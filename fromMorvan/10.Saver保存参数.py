from __future__ import print_function
import tensorflow as tf
import numpy as np

# 保存参数到文件中
# W = tf.Variable(tf.truncated_normal([2,3]), dtype=tf.float32, name='weights')
# b = tf.Variable(tf.truncated_normal([1,3]), dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#    sess.run(init)
#    print(sess.run(W))
#    print(sess.run(b))
#    save_path = saver.save(sess, "Saver/save_net.ckpt")
#    print("Save to path: ", save_path)


# 重新加载变量
W = tf.Variable(np.zeros([2,3]), dtype=tf.float32, name="weights")
b = tf.Variable(np.zeros([1,3]), dtype=tf.float32, name="biases")

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "Saver/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))