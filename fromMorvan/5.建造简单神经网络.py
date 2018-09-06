import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义每层的结构
def add_layer(input,in_size,out_size,active_function=None):
    Weights=tf.Variable(tf.random_normal([out_size,in_size]))
    biases=tf.Variable(tf.zeros([out_size,1])+0.1)

    Wx_plus_b=tf.matmul(Weights,input) + biases

    if active_function==None:
        outputs=Wx_plus_b
    else:
        outputs = active_function(Wx_plus_b)
    return outputs


#np.newaxis添加新轴,这时x_data.shape由(300,)->(1,300)
#此时，输入数据是每个样本只有一个特征的有300个样本的数据
x_data=np.linspace(-1,1,300, dtype=np.float32)[np.newaxis,:]
#print(x_data[np.newaxis,:].shape)     #(1,300)

noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)   #产生符合平均值为0,方差为0.05的正态分布的噪音误差
y_data=np.square(x_data)+noise       #生成带有噪音的数据的标签y_data


#使用两个占位符,其尺寸是[1,None]，None表示任何数字都行
xs=tf.placeholder(tf.float32,[1,None])
ys=tf.placeholder(tf.float32,[1,None])

#隐藏层的输入是数据x_data,采用10个神经单元，采用relu激活函数
hiden_layer=add_layer(xs,1,10,active_function=tf.nn.relu)
#输出层采用1个神经单元，不适用激活函数
pred=add_layer(hiden_layer,10,1,None)

#计算预测值误差，loss是带有两个placeholder的计算过程
#这里y_data,pred都是1*300的矩阵， reduction_indices=[1]先是按行求平均值，得到的是1*1的数组，
#外面再套一个reduce_sum,就得到这个纯数值
#需要注意的是reduce_sum,reduce_mean对行或者列计算返回的都是1*m的行向量
#loss=tf.reduce_sum(tf.reduce_mean(tf.square(ys-pred), reduction_indices=[1]),reduction_indices=[0])
loss=tf.reduce_mean(tf.square(ys-pred))
#print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

#train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#建立Session,初始化所有参数等
init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#因为这里采用一列是一个样本，所以样本x_data,y_data都是1*300,这样作为画图的坐标是不行的，必须是 行向量，即300*1,所以使用转置
ax.scatter(x_data.T,y_data.T)
plt.ion()          #使用ion使得画出第一个散点图后不消失，然后继续画后面的曲线图
plt.show()
#plt.pause(5)

#print(x_data.shape,y_data.shape,pred.shape)

for step in range(1000):
    #运行1000次梯度下降迭代
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step%50==0:
        try:
            # 消除上一次的曲线(因为第一次的上一次没有曲线，放在try中，没有上一次曲线，就抛出异常，啥也不做)
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(pred, feed_dict={xs: x_data})
        loss_value = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        #画出预测的去曲线图,注意这里也要矩阵的转置
        lines = ax.plot(x_data.T, prediction_value.T, 'r-', lw=5)

        #这里使用ax.text标出误差，有点问题，具体的可以参考Regression.py中，使用plt.cla()来清除画布
        # texts=ax.text(0.5, 0, 'Loss=%.4f' % loss_value, fontdict={'size': 20, 'color': 'red'})
        print(loss_value)
        plt.pause(0.5)

plt.ioff()
plt.show()