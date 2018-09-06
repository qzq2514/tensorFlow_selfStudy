import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#忽略一些无谓的警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#定义神经网络每层的结构
def add_layer(input,in_size,out_size,active_function=None):
    Weights=tf.Variable(tf.random_normal([out_size,in_size]))
    biases=tf.Variable(tf.zeros([out_size,1])+0.1)
    Wx_plus_b=tf.matmul(Weights,input) + biases
    if active_function==None:
        outputs=Wx_plus_b
    else:
        outputs = active_function(Wx_plus_b)
    return outputs


#xs是图片数据集,每张图片28*28,不规定样本的个数
#ys是图片的标签，0~9共10个标签结果
xs=tf.placeholder(tf.float32,[784,None])
ys=tf.placeholder(tf.float32,[10,None])


#建立一层的神经网络，每个样本784个特征，样本分类结果分为10个标签
#使用softmax作为分类
pred=add_layer(xs,784,10,active_function=tf.nn.softmax)

#使用交叉熵代价来计算误差(loss)，具体的交叉熵公式见"交叉熵代价函数.PNG"和交叉熵理论.PNG
cross_entropy=tf.reduce_sum(-tf.reduce_mean(ys*tf.log(pred), reduction_indices=[1]))

#使用梯度下降来最小化误差
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


def calcu_accuracy(v_xs,v_ys):
    global pred
    #使用测试数据集和已经训练好的神经网络来得到对应分类结果
    #softmax()每个样本的输出预测是10*1的向量，每个位置表示分类该样本是第i类的概率
    y_pred=sess.run(pred,feed_dict={xs: v_xs.T})
    #print(y_pred.shape)

    #每列的10个元素是某样本的预测值，argmax第二个参数0表示对列取最大值，即取每列的最大值作为预测结果
    #和tf.max取最大值不同，argmax是得到最大值所在的下标
    #返回的就是1*m的数组，m是样本数，其元素是0或1，分别表示样本分类错误或正确
    correct_pred=tf.equal(tf.argmax(y_pred,0),tf.argmax(v_ys.T,0))
    #print(tf.argmax(y_pred,0))

    #tf.cast将数据转化为对应的类型,correct_pred是1*m的数组,
    #上述的tf.equal中,y_pred，v_ys.T都是(10, 10000),然后argmax(,0)的0代表按列求最大值,但是返回的是(10000,)
    #即一维的向量都是m*1,而不是1*m,所以下面求精确度的时候，reduction_indices=0，对行求平均值，而不是等于1

    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32),reduction_indices=[0])
    result=sess.run(accuracy,feed_dict={xs: v_xs.T,ys:v_ys.T})
    return result



#开始梯度下降，迭代训练参数
for step in range(1000):
    #使用小批量梯度下降，每次用100个数据样本,mnist.train表示训练集
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs.T, ys: batch_ys.T})
    if step % 50 == 0:
        #使用测试集test.images计算精确度
        print(calcu_accuracy(mnist.test.images,mnist.test.labels))
