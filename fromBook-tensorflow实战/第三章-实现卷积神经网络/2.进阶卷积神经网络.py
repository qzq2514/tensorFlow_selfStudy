import tensorflow as tf
from cifar10 import cifar10_input,cifar10
import numpy as np
import  time

max_step=3000
batch_size=128
data_dir='cifar10/tmp/cifar10_data/cifar-10-batches-bin/cifar-10-batches-bin'

def variable_with_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev))
    if w1 is not  None:
        #这里tf.nn.l2_loss(var)得到权重矩阵var的l2范式-标量，w1也是标量
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        # 参数权重的l2正则化损失放在名为"losses"的集合中，该集合构成神经网络优化的总损失
        tf.add_to_collection("losses",weight_loss)
        return var

def getLoss(logits,labels):
    labels=tf.cast(labels,tf.int64)
    #sparse_softmax_cross_entropy_with_logits直接通过前向传播最后一层的输出logits-([batch_size,num_classes]大小)
    #和样本标签labels-([batch_size]大小)得到损失，函数内部会将labels变成[batch_size,num_classes]大小然后计算交叉损失
    #返回cross_entropy是[batch_size]大小，表示每个样本的交叉损失
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name="cross_entropy_per_example")

    cross_entropy_mean=tf.reduce_mean(cross_entropy,name="cross_entropy")
    tf.add_to_collection("losses",cross_entropy_mean)   #将样本平均交叉损失添加进losses集合，后面统一优化
    return tf.add_n(tf.get_collection("losses"),name="total_loss")   #返回要优化的所有损失的集合

#数据类型是24*24的三通道彩图
image_holder=tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder=tf.placeholder(tf.int32,[batch_size])



#---------------------网络结构开始

#第一个卷积层权重，5*5大小，通道数和图像通道数一样，w1=0.0表明不对第一层的卷积核参数进行正则化
weight1=variable_with_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
#第一层卷积，步长为1，SAME卷积，图像大小不变
kernel1=tf.nn.conv2d(image_holder,weight1,strides=[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
#使用tf.nn.bias_add给卷积结果添加偏置，也可以直接kernel1+bias1
conv1=tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
#对第一层的最终卷积结果进行最大池化，步长为2，大小为3，SAME卷积，最后输出结果大小为原图一半
pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#最大池化结果后添加一个lrn层(局部响应归一化层)
#lrn(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None):
# [a, b, c, d]分别表示一个batch中第a张图片在(b,c)位置的第d个通道,局部响应归一化公式如下
#sqr_sum[a, b, c, d]就表示在[a, b, c, d]位置的通道方向上前后depth_radius半径内的像素的平方和
#sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
#output根据输入图片、sqr_sum、偏置bias、系数alpha和指数beta得到输出矩阵
# output = input / (bias + alpha * sqr_sum) ** beta
lrn1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

weight2=variable_with_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
kernel2=tf.nn.conv2d(lrn1,weight2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
conv2=tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
lrn2=tf.nn.lrn(conv2,4,1.0,alpha=0.001/9.0,beta=0.75)
#第二个卷积过程的池化在局部响应归一化层之后
pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#建立全连接，将三维样本变成行向量形式
reshaped=tf.reshape(pool2,[batch_size,-1])
dim=reshaped.get_shape()[1].value    #得到每个样本维数
#第一个全连接层节点数为384,w1不为0则表示对该部分的权重采用l2正则化
weight3=variable_with_loss(shape=[dim,384],stddev=0.04,w1=0.004)
bias3=tf.Variable(tf.constant(0.1,shape=[384]))
local3=tf.nn.relu(tf.matmul(reshaped,weight3)+bias3)

#第二个全连接，使用192个节点
weight4=variable_with_loss(shape=[384,192],stddev=0.04,w1=0.004)
bias4=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(local3,weight4)+bias4)

#最后一层，正态分布的标准差设为上一个隐藏层节点数的倒数，并且不采用正则化
#这里最后一层没有做softmax,到后面使用函数进行直接计算softmax并同时计算损失
weight5=variable_with_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
bias5=tf.Variable(tf.constant(0.1,shape=[10]))
logits=tf.nn.relu(tf.matmul(local4,weight5)+bias5)

#-----------------网络结构到此结束

loss=getLoss(logits,label_holder)    #得到损失
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
#logits-[batch_size,num_class],label_holder-[num_class]
top_k_op=tf.nn.in_top_k(logits,label_holder,1)    #使用topK计算精确度，默认k=1

#下载数据集-其中源码我改过了
# cifar10.maybe_download_and_extract()
# print("cifar10数据下载成功~")


#distorted_input函数上采样，一张图片通过剪裁、旋转等变成多张图片
images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)        #训练数据
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)     #测试数据
print(images_train.shape)    #(128, 24, 24, 3)

#开始训练
#使用tf.InteractiveSession作为session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#不要使用tf.Session
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())

#启动线程
tf.train.start_queue_runners()


image_batch, label_batch = sess.run([images_train, labels_train])
# print(image_batch)
for step in range(max_step):
    start_time=time.time()
    # cifar10_input.distorted_inputs返回的是tensor类型，必须run一下才能得到包含真实值的变量
    image_batch, label_batch = sess.run([images_train, labels_train])
    # print(image_batch)
    _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration=time.time()-start_time
    if step%10==0:
        example_per_sec=batch_size/duration
        sec_per_batch=float(duration)
        format_str=("step:%d,loss=%.2f (%.1f example_per_sec;%.3f sec/batch)")
        print(format_str%(step,loss_value,example_per_sec,sec_per_batch))


num_example=10000
import math
num_iter=int(math.ceil(num_example/batch_size))
true_count=0
total_sample_count=num_iter*batch_size
step=0
while step<num_iter:
    image_batch,label_batch=sess.run(images_test,labels_test)
    predictions=sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count+=np.sum(predictions)
    step+=1

prediction=true_count/total_sample_count
print("precision: %.3f"%prediction)