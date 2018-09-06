import tensorflow as tf
import numpy as np

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#placeholder就相当于先申请一块对应类型的内存(占位符)
#imput1=tf.placeholder(tf.float32,[2,3])
imput1=tf.placeholder(tf.float32)
imput2=tf.placeholder(tf.float32)
mul=tf.multiply(imput1,imput2)    #定义数据相乘的功能

with tf.Session() as sess:
    #因为一开始mul中的imput1,imput2就是类似占位符的，然后具体运行run(mul)时
    #就要传入占位符具体代表的数值
    res=sess.run(mul,feed_dict={imput1:2.,imput2:5.})
    print(res)