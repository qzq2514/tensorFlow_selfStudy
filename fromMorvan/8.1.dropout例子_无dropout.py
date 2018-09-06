import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import LabelBinarizer

#加载手写数据集
digits=load_digits()
X=digits.data
y=digits.target

y=LabelBinarizer().fit_transform(y)#标签二值化,比如标签是数字3那么就得到其输出向量是[0 0 0 1 0 0 0 0 0 0]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)#将数据集按照7:3划分为训练集和测试集

#定义每一层网络的结构
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([out_size,in_size]))
    biases=tf.Variable(tf.zeros([out_size,1])+0.1)
    Wx_plus_b=tf.matmul(Weights,inputs)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    #tf.summary.histogram(layer_name+'/outpus',outputs)
    return outputs

xs=tf.placeholder(tf.float32,[64,None])   #图片数据是8*8图片,每个像素点是一个特征，则输入有64个特征
ys=tf.placeholder(tf.float32,[10,None])

#定义中间层l1(50个单元)和输出层(10个单元)
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
pred=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

#定义误差
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),reduction_indices=[1]),reduction_indices=[0])
#cross_entropy= tf.reduce_sum(-tf.reduce_mean(ys*tf.log(pred),reduction_indices=[1]))
cross_entropy=-tf.reduce_mean(ys*tf.log(pred))
tf.summary.scalar("loss",cross_entropy)      #记录损失代价值
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
merged=tf.summary.merge_all()    #定义整合的过程，以便下面通过训练集训练出来的数据可以用到测试集计算误差

sess.run(tf.global_variables_initializer())

#使用tensorboard记录训练集和测试集的优化过程
train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_writer=tf.summary.FileWriter("logs/test",sess.graph)

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train.T,ys:y_train.T})
    if i%50 ==0:   #每50步记录训练集和测试集误差
        train_res=sess.run(merged,feed_dict={xs:X_train.T,ys:y_train.T})   #计算
        test_res = sess.run(merged, feed_dict={xs: X_test.T, ys: y_test.T})
        train_writer.add_summary(train_res,i)   #每50步得到一次训练集和测试集误差
        test_writer.add_summary(test_res,i)