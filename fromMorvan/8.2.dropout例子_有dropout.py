import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import LabelBinarizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#加载手写数据集
digits=load_digits()
X=digits.data
y=digits.target

y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#定义每一层网络的结构
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([out_size,in_size]))
    biases=tf.Variable(tf.zeros([out_size,1])+0.1)
    Wx_plus_b=tf.matmul(Weights,inputs)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)  #采用dropout
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs


keep_prob=tf.placeholder(tf.float32)        #定义保留概率
xs=tf.placeholder(tf.float32,[64,None])   #图片数据是8*8图片,每个像素点是一个特征，则输入有64个特征
ys=tf.placeholder(tf.float32,[10,None])

#定义中间层l1(50个单元)和输出层(10个单元)
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
pred=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

#cross_entropy= tf.reduce_sum(-tf.reduce_mean(ys*tf.log(pred),reduction_indices=[1]))
cross_entropy=-tf.reduce_mean(ys*tf.log(pred))
print(pred)
tf.summary.scalar("myLoss",cross_entropy)      #记录损失代价值
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
sess=tf.Session()
merged=tf.summary.merge_all()    #定义整合的过程，以便下面通过训练集训练出来的数据可以用到测试集计算误差

sess.run(tf.global_variables_initializer())

#使用tensorboard记录训练集和测试集的优化过程
train_writer=tf.summary.FileWriter("logs/myTrain",sess.graph)
test_writer=tf.summary.FileWriter("logs/myTest",sess.graph)

for i in range(500):
    #采用keep_prob=0.6
    sess.run(train_step,feed_dict={xs:X_train.T,ys:y_train.T,keep_prob:0.6})
    if i%50 ==0:   #每50步记录训练集和测试集误差
        train_res=sess.run(merged,feed_dict={xs:X_train.T,ys:y_train.T,keep_prob:1})   #计算
        test_res = sess.run(merged, feed_dict={xs: X_test.T, ys: y_test.T,keep_prob:1})
        train_writer.add_summary(train_res,i)   #每50步得到一次训练集和测试集误差
        test_writer.add_summary(test_res,i)