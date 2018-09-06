import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#定义过滤器，尺寸为[5,5,3,16],其中5x5是过滤器长宽,3是上一层的输入的结点矩阵深度,16是过滤器的数量，也是上一层输入和过滤器卷积操作后生成的即结点矩阵的深度
filter_weights=tf.get_variable(name="weights",shape=[5,5,3,16]
                               ,initializer=tf.truncated_normal_initializer(stddev=0.1))    #共有16个5x5x3的过滤器

#偏置只和输出的结点矩阵深度有关,相同层的不同卷积操作的偏置都是一样的
biases=tf.get_variable(name="biases",shape=[16],initializer=tf.constant_initializer(0.1))


#输入数据为2*28*28*3，其中2表示样本数，即这里有两张图片,28*28表示每张图片的长宽,3是图片的深度(也就是深度)
input_data=tf.get_variable(name="input",shape=[2,28,28,3],initializer=tf.truncated_normal_initializer(0.1))



#tf.nn.conv2d实现卷积操作,前两个参数分别是输入的图片数据和过滤器，第三个参数是步长，虽然是4维的，但是
conv=tf.nn.conv2d(input=input_data,filter=filter_weights,strides=[1,2,2,1,],padding="SAME")#VALID:(2, 12, 12, 16)
                                                                                           #SAME:(2, 14, 14, 16)


bias=tf.nn.bias_add(conv,biases)     #(2, 14, 14, 16)

#测试tf.nn.relu
# a = tf.Variable(np.arange(-60,60).reshape(2,3,4,5))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     b = tf.nn.relu(a)
#     print(sess.run(b))

actived_conv=tf.nn.relu(bias)      #(2, 12, 12, 16)--激活函数不会改变张量维度

#第一个参数是输入结点，第二个参数是过滤器大小(只有表示长宽的2，3维),后面两个参数依次是步长和填充方式
#池化层的核只是求方框内的最大值(max_pool)或者平均值(avg_pool),不需要像卷积核一样需要参数
pool=tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")   #VALID:(2, 6, 6, 16),Same:(2, 7, 7, 16)
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer()))
    print(bias)
    print(pool)

