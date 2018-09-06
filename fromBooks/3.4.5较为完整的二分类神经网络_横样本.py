import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#最后的决策边界不准确


batch_size=8   #训练集batch大小


#神经元数依次为2,3,1的神经网络
W1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
W2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))


#定义placeholder
x=tf.placeholder(tf.float32,shape=(None,2),name="X-input")
y_=tf.placeholder(tf.float32,shape=(None,1),name="Y-input")


#定义前向传播
a=tf.matmul(x,W1)
y=tf.matmul(a,W2)

# print(tf.transpose(y))
#使用交叉熵定义损失函数，先用clip_by_value将预测值归一化到0-1
cross_entropy=-tf.reduce_mean(tf.transpose(y_)*tf.log(tf.clip_by_value(tf.transpose(y),1e-10,1.0)))

train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm=RandomState(1)#设置随机变量，1是种子，种子一样，随机数就是一样的
data_size=128
X=rdm.rand(data_size,2)  #随机产生128个特征数为2的输入集


#规定两个特征和x1+x2<1则标签记为1(正类)
#要注意，因为遍历二维数组，每一行为一个元素，二原来X是2*128，所以要转置，保证每一行为一个样本
Y=[int(x1+x2<1) for x1,x2 in X] #这里产生1*128的list（小技巧：[[int(x1+x2<1)] for x1,x2 in X.T]产生128*1的list）
color=Y
Y=np.array(Y)
Y=Y.reshape(data_size,1)
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    # print(X[:,0])
    # print(Y.reshape(1,128).shape)
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=color)  #除去size为1的维度:
    plt.show()
    print("前：",sess.run(W1))
    print("前：",sess.run(W2))

    step=5000

    for i in range(step):
        start=(i*batch_size)%data_size
        end=min(start+batch_size,data_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if (i+1)%500==0:
         curCrossEntropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})  #计算在所有数据样本上的交叉熵代价值
         print("当前第%d步交叉熵：%f\n"%(i+1,curCrossEntropy))

    print("后：",sess.run(W1))
    print("后：",sess.run(W2))
    print(sess.run(W2))

    testSize = 100
    y_out = np.zeros([testSize, testSize])
    xx1 = np.linspace(0, 1, testSize)
    xx2 = np.linspace(1, 0, testSize)
    X1, X2 = np.meshgrid(xx1, xx2)

    xtemp1 = X1[0,:][np.newaxis,:]
    xtemp2 = X2[0,:][np.newaxis,:]  # testSize - 0 - 1
    this_X = np.vstack((xtemp1, xtemp2))

    print(this_X)
    print(X2)
    print(X1[1,:][np.newaxis,:].shape)
    for i in range(testSize):
        xtemp1 = X1[i,:][np.newaxis,:]
        xtemp2 = X2[i,:][np.newaxis,:]
        this_X = np.vstack((xtemp1, xtemp2))
        temp=sess.run(y, feed_dict={x: this_X.T})
        y_out[i, :]=temp.T   #原来temp.shape是(100,1)不能直接赋值给y_out[i, :]-（100，）但是(1，100)的矩阵可以直接赋值给（100，），所以要temp.T

    print("前y_out:", y_out)
    tempY = y_out
    y_out = np.where(tempY < 1, y_out, 1)
    y_out = np.where(tempY > 1, y_out, 0)

    y_out[tempY >1]=1
    y_out[tempY < 1] = 0
    print("后y_out:", y_out)
    plt.contour(X1, X2, y_out, 5)
    plt.show()