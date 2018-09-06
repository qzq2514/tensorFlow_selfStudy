import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess=tf.Session()
sess.run(tf.global_variables_initializer())


#产生标准差(stddev)=0.1的正太分布的2*3数组
initial=tf.truncated_normal([2,3],stddev=0.1)
print(sess.run(initial))

#产生值全是0.1的2*3的数组
initial=tf.constant(0.1,shape=[2,3])
print(sess.run(initial))