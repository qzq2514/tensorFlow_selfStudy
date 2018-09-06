import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_weight(shape,lam):
    var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)

    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lam)(var))  #正则化项，单个数值，最后的损失值的一部分

    return var


x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
batch_size=8

layer_dimension=[2,10,10,10,1]

n_layers=len(layer_dimension)

#保存当前层的结点，一开始就是输入层
cur_layer=x
#当前层的结点数
in_dimension=layer_dimension[0]

#通过循环生成一个五层的神经网络
for i in range(1,n_layers):
    out_dimension=layer_dimension[i]
    weight=get_weight([in_dimension,out_dimension],0.001)
    bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))   #????

    cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    in_dimension=out_dimension   #开始为下一层网络准备


#循环结束，cur_layer最后表示整个网络的输出
mes_loss=tf.reduce_mean(tf.square(y_-cur_layer))   #损失值的第一项

tf.add_to_collection("losses",mes_loss)
loss=tf.add_n(tf.get_collection("losses"))   #tf.add_n：把一个列表的东西都依次加起来
train_op=tf.train.AdamOptimizer(0.01).minimize(loss)

#下面的程序和之前的(4.2.2)是一样的，就是训练保证损失值最小化

data_size=128
X=np.random.normal(size=(data_size,2))
# Y=[[2*x1+x2*x2+np.random.rand()/10-0.05] for x1,x2 in X]
Y=[[2*x1+x2*x2] for x1,x2 in X]
Y=Y+np.random.normal(size=(data_size,1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps=5000

    for i in range(steps):
        start=(i*batch_size)%data_size
        end=min(start+batch_size,data_size)

        sess.run(train_op,feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(loss,feed_dict={x:X,y_:Y}))
