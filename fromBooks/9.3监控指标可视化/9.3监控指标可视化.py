import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

Summary_dir="log"
Batch_size=100
Train_steps=1000


#生成变量监控信息并定义生成监控信息日志的操作，其中var是需要记录的张量，name是可视化结果中显示的图表名称，一般与变量名一致

def variable_summaries(var,name):
    tf.summary.histogram(name,var)   #记录张量中元素的取值分布

    mean=tf.reduce_mean(var)     #没有指定维度求均值，则表明对其中全部元素进行求均值,最终返回一个单个元素的张量
    tf.summary.scalar("mean/"+name,mean)     #显示标量信息,这里"/"前的"mean"是命名空间，后面相同命名空间的变量会放在一起
    stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))    #求的整体元素的标准差
    tf.summary.scalar("stddev/"+name,stddev)




def nn_layer(inputTensor,input_dim,output_dim,layerName,activeFunction=tf.nn.relu):
    #将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope(layerName):
        with tf.name_scope("weights"):
            weights=tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
            variable_summaries(weights,"weights")             #记录权重和偏执
        with tf.name_scope("biases"):
            biases=tf.Variable(tf.constant(0.0,shape=[output_dim]))
            variable_summaries(biases,"biases")
        with tf.name_scope("Wx_plus_b"):
            preactivation=tf.matmul(inputTensor,weights)+biases          #记录激活函数之前的节点的取值分布
            tf.summary.histogram("pre_activations",preactivation)
            #在tensorboard的计算图可以看到，在激活函数relu之前，"layer1/pre_activations"变量分布正负都有，
            #而在relu之后，"layer1/activation"上的变量分布都是大于0的
            activations=activeFunction(preactivation,name="activation")
            tf.summary.histogram("activation",activations)
    return activations




def main(_):
    mnist=input_data.read_data_sets("../MNIST_data",one_hot=True)
    with tf.name_scope("input"):
        x=tf.placeholder(tf.float32,[None,784],name="x-input")
        y_=tf.placeholder(tf.float32,[None,10],name="y-input")


    with tf.name_scope("input-reshape"):
        image_shaped_input=tf.reshape(x,[-1,28,28,1])     #-1表示这个维度我们不用自己指定，函数会自动计算，但是最多只能有一个-1
        tf.summary.image("inputImage",image_shaped_input,10)  #展示训练过程中记录的图像,最大展示10张图片

    hidden1=nn_layer(x,784,500,"layer1")    #第一层:隐藏层
    y=nn_layer(hidden1,500,10,"layer2",activeFunction=tf.identity)  #y=tf.identity(x)就相当于赋值操作，但是相比较于y=x,tf.identity会在计算图中增加一个操作节点

    with tf.name_scope("cross_entropy"):
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    tf.summary.scalar("crooss_entropy",cross_entropy)

    with tf.name_scope("train"):
        train_step=tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)



    with tf.name_scope("accuracy_scope"):
        with tf.name_scope("correct_prediction"):
            correct_prediction=tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
        with tf.name_scope("accuracy"):
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar("accuracy",accuracy)

    merged=tf.summary.merge_all()    #像之前的tf.summary.scalar，tf.summary.image，tf.summary.histogram都是需要通过sess.run来调用执行
                                    #一一调用显得过于繁琐，就可以将所有的操作合并，然后通过后面sess.run(merged）整个运行所有的监控指标的记录操作

    with tf.Session() as sess:
        summary_writer=tf.summary.FileWriter("log",sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(Train_steps):
            xs,ys=mnist.train.next_batch(Batch_size)
            summary,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys})   #进行监控指标的记录，并开始训练
            summary_writer.add_summary(summary,i)  #将所有监控指标信息summary写入文件，然后既可以在tensorboard中查看运行信息，生成的图中有每一步i的时的变量信息

        summary_writer.close()


if __name__ == '__main__':
    tf.app.run()

