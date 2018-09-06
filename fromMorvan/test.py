import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#注意！！！在卷积神经网络中，最好就用横向的样例，这里不要像之前的用纵向的样例
#因为在你输入图片的时候，图片的像素被转置后，即batch_xs.T后，那么图片就反转了，在进行reshape后，会出现问题
#至于之前在"7.实现简单手写数字分类.py"，mnist得到的就是横向的样例,我们虽然采用纵向样例，但是并没有reshape组成图片，
#并不会影响最后的图片识别
#计算预测准确度
def calcu_accuracy(v_xs,v_ys):
    global pred
    y_pred=sess.run(pred,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})  #计算预测值
    #print("----------",y_pred.shape)
    correct_pred=tf.equal(tf.argmax(y_pred,1),tf.argmax(v_ys,1))  #计算预测正确的个数
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    res=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return res


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)  #根据shape产生标准差(stddev)=0.1的正太分布的数组
    return tf.Variable(initial)

def biases_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#进行二维图片(灰度图)的卷积,x是灰度图，W是二维的卷积过滤器
def conv2d(x,W):
    #使用tf自带的卷积:conv2d.其中strides参数必须是[1,a,b,1]形式，a,b分别是水平和竖直方向的步长
    #padding是填充方式:1.SAME原图和输出图大小相等,2,Valid,输出图片会缩小
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pooling_2x2(x):
    #ksize是池化的时候的卷积核大小,必须是[1,a,b,1]形式，a,b分别是核的宽高
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

xs=tf.placeholder(tf.float32,[None,784])/255.   #图片28*28
ys=tf.placeholder(tf.float32,[None,10])   #标签
keep_prob=tf.placeholder(tf.float32)
#shape:[batch, height, width, channels]
x_image=tf.reshape(xs,[-1,28,28,1])

#定义卷积层的过滤器，这里是其大小5*5*1，因为输入是灰度图只有1个通道，所以最后是*1,
#而32是过滤器的个数，其实也是输出图片的通道数
W__conv1=weight_variable([5,5,1,32])
#偏置参数b个数等于该层的过滤器个数
b_conv1=biases_variable([32])
#得到输出,即28*28*1的图片经过32个5*5*1的过滤器卷积(即conv2d操作)后分别加上偏置
h_conv1=tf.nn.relu(conv2d(x_image,W__conv1)+b_conv1)  #输出的图片是28*28*32
h_pool1=max_pooling_2x2(h_conv1) #卷积后的图片最大池化,池化的步长和核边长都是2，大小减半，得到14*14*32的图片(池化不会改变通道数)


#上一层得到的图片是14*14*32,
W__conv2=weight_variable([5,5,32,64])
b_conv2=biases_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W__conv2)+b_conv2)   #输出14*14*64
h_pool2=max_pooling_2x2(h_conv2)                    #输出7*7*64大小的图片


# #全连接层1
W_fc1=weight_variable([7*7*64,1024])   #得到7*7*64(即图片的总像素数)到1024个单元的全连接层;
b_fc1=biases_variable([1024])
#将图片的所有像素展开(-1表示维度不变，这里等于样本的个数)，即([n_sampls,7,7,64]->>[n_sampls,7*7*64])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)  #像之前的神经网络一样，进行隐藏层的计算，并使用relu激活函数(输出[n_sampls,1024])
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)    #添加dropout避免过拟合

# #全连接层2
W_fc2=weight_variable([1024,10])   #连接上层的1024单元和输出层的10个分类单元
b_fc2=biases_variable([10])
pred=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)   #(输出[n_sampls,10])


#训练准备
cross_entropy=-tf.reduce_mean((ys)*tf.log(pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  #Adam最小化代价


# print(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# #开始训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #每次取训练集的100个数据进行小批量训练(batch_xs:[100,784],batch_ys:[100,10])
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(calcu_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))