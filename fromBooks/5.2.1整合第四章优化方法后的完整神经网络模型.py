import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node=784    #输入层结点数
output_node=10    #输出层结点数

hidden_node=500    #该网络只有一个隐藏层，结点数是500

batch_size=100  #批量梯度下降的批量尺寸

learning_rate_base=0.8   #初始化的学习率
learning_rate_decay=0.99 #学习率的衰减率

regularization_lambda=0.0001   #正则化系数
training_step=30000    #训练迭代次数
moving_average_rate=0.99    #滑动平均衰减率


#avg_class:滑动平均类别
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2  #返回神经网络输出
    else:    #使用滑动平均中的参数
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))
                                                +avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)



def train(mnist):
    x=tf.placeholder(tf.float32,[None,input_node],name="x_input")
    y_=tf.placeholder(tf.float16,[None,output_node],name="y_input")


    weights1=tf.Variable(
        #这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
        tf.truncated_normal([input_node,hidden_node],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[1,hidden_node]))

    weights2 = tf.Variable(
        # 这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
        tf.truncated_normal([hidden_node,output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[1,output_node ]))

    #这里的是不用平均滑动模型
    y=inference(x,None,weights1,biases1,weights2,biases2)


    #训练轮数,这是不断变化的，但不是要训练的参数
    global_step=tf.Variable(0,trainable=False)
    variable_average=tf.train.ExponentialMovingAverage(moving_average_rate,global_step )

    #将所有可训练的参数添加进滑动平均模型（注意：tf.Variable(）中默认trainable=True,默认都是可训练的）
    variable_average_op=variable_average.apply(tf.trainable_variables())

    #使用平均滑动模型
    average_y=inference(x,variable_average,weights1,biases1,weights2,biases2)

    #对预测输出y_每行求最大值,y是[m，output_node],y_-[m，output_node],arg_max行求和后，变成[m,]
    #必须显示地对参数进行赋值，logits=需要softmax的值,labels=是原样本的标签
    #返回变量尺寸为(m,)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cross_entropy_mean=tf.reduce_mean((cross_entropy))  #交叉熵求平均值


    regularizer=tf.contrib.layers.l2_regularizer(regularization_lambda)  #定义正则化
    regularization=regularizer(weights1)+regularizer(weights2)  #正则化只需要对选中参数进行正则化，无需对偏置记性呢正则化

    loss=cross_entropy_mean+regularization   #交叉熵平均值和正则化都是单一的数值型，直接相加


    #learning_rate=learning_rate_base*learning_rate_decay^(global_step/decay_step),其中
    #decay_step是第三个参数，即mnist.train.num_examples/batch_size
    #还有个参数decay_step表示每decay_step步更新衰减一次学习率
    #global_step是当前的迭代轮数，不断变化的
    learning_rate=tf.train.exponential_decay(
        learning_rate_base,     #基础的学习率
        global_step,
        mnist.train.num_examples/batch_size,    #decay_step，更新速度，即每优化这么多次更新一次学习率(这个参数是不定的)
        learning_rate_decay,
    )

    #这里一定要加上global_step=global_step，保证变量global_step不断变化，然后进一步传进exponential_decay函数中
    #之前GradientDescentOptimizer中的学习率参数都是固定的，像0.01之类，这里则传入了变量，保证学习率不断变化
    #global_step自增,在staircase=True时，每到global_step%decay_step==0时更新一次学习率
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)


    #同时完成反向传播的参数训练模型，得到新的参数，这是又要滑动平均模型，利用新参数值和旧值更新每一个参数的滑动平均值
    #这里使用control_dependencies函数保证一次完成多个操作，其和train_op=tf.group(train_step,variable_average_op)是等价的
    #注意到这里的两个步骤都需要用到global_step，在train_step中保证global_step
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op=tf.no_op(name="train")    #tf.no_op啥也不做，纯属凑数


    #tf.argmax(average_y,1)求每个样本的预测值(尺寸为:[batch_size,1]),这里返回True,False矩阵
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accury=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #tf.cast将bool型矩阵转化为float型


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        validate_feed={x:mnist.validation.images,
                       y_:mnist.validation.labels}   #验证集

        test_feed = {x: mnist.test.images,
                         y_: mnist.test.labels}    #测试集


        for i in range(training_step):
            if i %1000==0:    #每1000步打印一次结果
                #验证集的数据不多，可以在整个验证集上进行验证
                validate_accu=sess.run(accury,feed_dict=validate_feed)
                print("在%d轮迭代后，验证集的精确度为%g"%(i,validate_accu))  #%g:根据值的大小采用%e或%f，但最多保留6位有效数字

            xs,ys=mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #全部训练完后，在测试集上看下模型精确度
        test_accu=sess.run(accury,feed_dict=test_feed)
        print("在%d轮迭代后，验证集的精确度为%g" % (training_step, test_accu))


def main(argv=None):
    mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
    train(mnist)


#tensorflow提供一个主程序入口，tf.app.run()会调用上面定义的main()函数
if __name__=="__main__":
    tf.app.run()


