import tensorflow as tf
import numpy as np


#shape:(4,)即长度为4的行向量
arr1=tf.Variable([[1.,2,3,4],
                 [5,6,7,8,],
                  [9,10,11,12]])

arr2=tf.Variable([[1.,4,10,4],
                 [10,4,6,8,],
                  [9,2,8,45]])



sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tf.maximum(arr1,arr2)))#求两个同维度的数组最大值　

print(sess.run(tf.argmax(arr2)))      #默认求列最大值
print(sess.run(tf.argmax(arr2,0)))      #同上

res=sess.run(tf.equal(arr1,arr2))
print(res)  #比较相同大小的数据个位置元素是否相等,每个元素是True或者False

print(sess.run(tf.cast(res,tf.float32)))#将True变1.，False变0