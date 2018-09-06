import tensorflow as tf
import numpy as np

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义变量
var1=tf.Variable(0,name='var1')
print(var1.name)

#定义常量
cons1=tf.constant(1)

#相加产生新值
new_value=tf.add(var1,cons1)
#定义更新过程，new_value->var1
update=tf.assign(var1,new_value)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    #
    sess.run(init)
    for _ in range(3):  #进行三次更新过程
        sess.run(update)
        print(sess.run(var1))
