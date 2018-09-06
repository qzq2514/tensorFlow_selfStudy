import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size=8

#简单回归问题下的神经网络，两个输入，一个输出
x=tf.placeholder(tf.float32,shape=(None,2),name="x_input")
y_=tf.placeholder(tf.float32,shape=(None,1),name="y_input")


w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
b=tf.Variable(tf.constant(0.0))
y=tf.matmul(x,w1)

#少生产一个产品，损失10元利润
loss_less=10
#多生产一个产品，多付1元成本
loss_more=1

#自定义损失函数，多生产和少生产的损失权重不一样，这里不能简单使用均方误差
#tf.where替代了原来的tf.select,第一个参数真返回第二个参数，否则返回第三个参数
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

#生成模拟数据集
rdm=RandomState(1)
data_size=128

X=rdm.rand(data_size,2)    #产生随机数据
#假定y=x1+x2+2然后再添加一些-0.05~0.05的噪声
Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

Data=np.hstack((X,Y))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps=5000
    for i in range(steps):

        # 1.不随机小批量梯度下降-不打乱原数据，每条数据都会被用作训练
        # start=(i*batch_size)%data_size
        # end=min(start+batch_size,data_size)
        # sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        #2.随机小批量梯度下降-随机打乱数据，每次取前batch_size条数据都是不一样的
        ind=np.arange(data_size,dtype=np.int32)
        np.random.shuffle(ind)
        ind=ind[0:batch_size]
        sess.run(train_step, feed_dict={x: Data[ind,0:2],y_: Data[ind,2][:,np.newaxis]})
    print(sess.run(w1),sess.run(b))

#如果这里没有偏置项，并且设定y=x1+x2，那么最终得到的参数w1大于等于[1.02186108,1.03858542],因为预测少的话损失更大，所以为了
#最小化损失，可以稍微多预测下。
#若将loss_more=100，则预测多时损失更大，这样最后w2=[0.95617229,0.98344052],即偏向于少预测
