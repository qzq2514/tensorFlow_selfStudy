import tensorflow as tf
import numpy as np

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#生产随机数据
#astype实现变量类型转换
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3


#建立tensorflow结构
#random_uniform([1])返回1*1的矩阵，产生于low和high之间，产生的值是均匀分布的
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
Bias=tf.Variable(tf.zeros([1]))
#print(Weights,Bias)

#根据参数得到预测的模型
y=Weights*x_data+Bias

#计算预测误差平均值
loss=tf.reduce_mean(tf.square(y-y_data))

#创建学习率alpha=0.5的梯度下降优化器
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化所有参数
init=tf.global_variables_initializer()

######到此，tensorflow的结构全部定义结束


#每次训练就是一次回话
sess=tf.Session()
sess.run(init)           #激活initational

#Weights,Bias都是张量，直接输出得不到值，必须使用sess.run()输出
print(sess.run(Weights),sess.run(Bias))



for step in range(200):
    #进行训练,每次训练，参数Weights,Bias都会被优化
    sess.run(train)

    if step%20==0:
        print(step,sess.run(Weights),sess.run(Bias))

#在for循环中，每次参数都被训练，到最后，Weights,Bias会越来越接近真实的结果0.1和0.3