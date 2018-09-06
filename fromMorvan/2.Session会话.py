import tensorflow as tf
import numpy as np

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#constant定义常量
mat1=tf.constant([[3,3]])
mat2=tf.constant([[2],
                  [2]])
prod=tf.matmul(mat1,mat2)   #矩阵乘法,相当于np.dot(mat1,mat2)


#方法1
sess=tf.Session()
print(prod)
#不能直接打印prod,其是个Tensor类型，要用会话运行下
print(sess.run(prod))
sess.close()

#方法2
#就像打开文件一样，使用with as结构，之后会自动调用close方法
with tf.Session() as sess:
    print(sess.run(prod))