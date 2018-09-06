import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_layer(input,in_size,out_size,n_layer,active_function=None):
    layer_name="qzq_7.19layer%s"% n_layer          #定义每层神经网络的名字
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights=tf.Variable(tf.random_normal([out_size,in_size]),name="W")
            tf.summary.histogram("Weights",Weights)   #查看某个变量的历史变化,第一个参数是在图中显示的名称
        with tf.name_scope('Biases'):
            biases=tf.Variable(tf.zeros([out_size,1])+0.1,name='b')
            tf.summary.histogram("Biases", biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(Weights,input) + biases
        if active_function==None:
            outputs=Wx_plus_b
        else:
            outputs = active_function(Wx_plus_b)
        return outputs


x_data=np.linspace(-1,1,300, dtype=np.float32)[np.newaxis,:]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)+noise

#构造输入层，name_scope建立一个可折叠打开的框框,这里为输入层建立命名空间，xs,ys都属于输入层
with tf.name_scope('qzq_7.19inputs'):
    #name="x_input"为某数据单元命名，都会显示在最后的tensorboard中
    xs=tf.placeholder(tf.float32,[1,None],name="x_input")
    ys=tf.placeholder(tf.float32,[1,None],name="y_input")


hiden_layer=add_layer(xs,1,10,n_layer=1,active_function=tf.nn.relu)
pred=add_layer(hiden_layer,10,1,n_layer=2,active_function=None)

with tf.name_scope('Loss'):
    loss=tf.reduce_sum(tf.reduce_mean(tf.square(ys-pred), reduction_indices=[1]),name='loss')
    tf.summary.scalar("loss",loss)       #loss不像Weights和biases一样是矩阵，他是单变量，查看他的变化，要用tf.summary.scalar
with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#建立Session,初始化所有参数等
sess = tf.Session()

#合并所有的summary
merge=tf.summary.merge_all()
#将Graph生成到logs/目录下，会自动命名
#查看该图:
#1. 则先进入logs所在的目录
#2.windows下:tensorboard --logdir=logs    mac两种方法步骤如下：
#方法一：
# (1).在python环境下：pip show tensorflow

#(2).cd "site-packages路径"（本机是/anaconda3/envs/Python3_Ana/lib/python3.6/site-packages）

#(3).cd tensorboard
#(4).python main.py --logdir=logs文件夹目录（本机是/Users/qzq2514/Code/WorkSpace/Pycharm/TensorFlow/TensorflowLearning/fromMorvan/logs）
#(5).在浏览器输入指定的网址

#方法二：
#tensorboard --logdir=logs/（这里在logs文件夹的上层目录里运行，比如这里在fromMorvan文件夹里打开终端运行这句话）
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(tf.global_variables_initializer())


#开始训练神经网络
for step in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        result=sess.run(merge,{xs: x_data, ys: y_data})   #每隔50步整合一次数据
        writer.add_summary(result,step)         #将这些整合信息再添加到tensorboard中
