import tensorflow as tf
import numpy as np


#shape:(4,)即长度为4的行向量
arr=tf.Variable([[1.,2,3,4],
                 [5,6,7,8,]])

sumValue=tf.reduce_sum(arr)         #求所有元素的和-----36
sumValue0=tf.reduce_sum(arr,0)      #列求和----[  6.   8.  10.  12.](1*4)
sumValue1=tf.reduce_sum(arr,1)     #行求和-----[ 10.  26.](1*2)

meanValue=tf.reduce_mean(arr)       #求所有元素的平均值--4.5(当arr全是整数1~8,那么平均值是4)
meanValue0=tf.reduce_mean(arr,0)    #对列求平均值[ 3.  4.  5.  6.](1*4)
meanValue1=tf.reduce_mean(arr,reduction_indices=[1])    #对行求平均值[ 2.5  6.5](1*2)

step11=tf.reduce_mean(arr, reduction_indices=[1])   #[ 2.5  6.5]
step12=tf.reduce_sum(step11,reduction_indices=[0])   #9.0

step21=tf.reduce_sum(arr, reduction_indices=[1])   #[ 10.  26.]
step22=tf.reduce_mean(step21)                       #18.0

sess=tf.Session()
sess.run(tf.global_variables_initializer())

res=sess.run(step22)
print(res)