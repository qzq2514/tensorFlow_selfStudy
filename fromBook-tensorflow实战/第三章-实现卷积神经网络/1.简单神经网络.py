from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)     #一定要将初始化包含在Variable变量中，因为这些是后面要优化的参数，是变量类型的
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#tensorflow中池化和卷积函数的strides参数表示步长，目前仅第2，3个参数有效，这里[1,2,2,1]表示长宽方向的步长都是2
#SAME卷积:ceil(输入高/高步长)*ceil(输入宽/宽步长)    --SAME卷积下输出尺寸与卷积核大小无关，仅仅与步长有关(步长为1时输入尺寸不变)）
#VALID卷积:ceil((输入高-卷积核高+1)/高步长)*ceil((输入宽-卷积核宽+1)/宽步长)   --VALID卷积输出尺寸与卷积核大小和步长都有关
def conv2d(x,W):
    return tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])    #将向量形式的输入样本变为2D的图像样本，-1表示图像样本数(不固定，自动计算)，
                                      #28*28先后表示行数列数(即图像高宽)，1表示通道数
W_conv1=weight_variable([5,5,1,32])    #第一个卷积层-卷积核尺寸为5*5,一个颜色通道(灰度图，一个通道就够),32个不同卷积核
b_conv1=bias_variable([32])       #一个卷积核对应一个权重偏置
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)  #relu作为激活函数
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,32,64])   #前一步通过32个卷积核卷积后得到32通道的“图像”
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#这里卷积都采用"SAME"卷积且步长都是1*1,所以经过卷积后尺寸是不变的，
#唯一可能改变图像尺寸的是池化，这里池化也是采用"SAME"，但是步长是2*2,所以通过两次池化后，图像尺寸变成7*7
#同时最后一次有64个卷积核，所以到这里的全连接，输入的一张图像尺寸为7*7*64,这里全连接节点数定义为1024
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool_flat=tf.reshape(h_pool2,[-1,7*7*64])   #每个样本展开成一个行向量
#h_pool_flat:[样本数,7*7*64],W_fc1:[7*7*64,1024]---->tf.matmul矩阵乘法后得到[样本数,1024]，在通过广播加上b_fc1-[1024]）
h_fc1=tf.nn.relu(tf.matmul(h_pool_flat,W_fc1)+b_fc1)

#为了防止过拟合，第一个全连接层到第二个全连接层中采用dropout
k_prop=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=k_prop)

W_fc2=weight_variable([1024,10])  #最后一层10个节点代表0-9十个分类结果
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  #至此得到分类结果存放在h_fc2中-----shape=(?, 10)

#计算损失，reduce_sum内的reduction_indices=[1]表示对列求和，得到的每一行有一个数字，代表一个样本的损失
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#求精确度，同样对于行样本，进行列方向的argmax，得到每行跨列的最大值下标
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
# correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #计算精确度，这里最好进行类型转换，因为之前tf.equal得到bool类型


mnist=input_data.read_data_sets("../../MNIST_data",one_hot=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(y_conv,y_)
    for i in range(20000):
        #每次得到50个样本，batch[0]是数据-[50,784]，batch[1]是标签-[50,10]
        batch=mnist.train.next_batch(50)
        if i%100==0:
            #eval()其实就是Session.run() 的另外一种写法
            #测试精度时不用dropout
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],k_prop:1.0})
            # train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],k_prop:1.0})
            print("step %d,training accuracy:%g"%(i,train_accuracy))
        train_op.run(feed_dict={x:batch[0],y_:batch[1],k_prop:0.5})
    #最后得到测试集精度
    correctPred=correct_prediction.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,k_prop:1.0})
    print(correctPred)
# print(mnist.train.num_examples)