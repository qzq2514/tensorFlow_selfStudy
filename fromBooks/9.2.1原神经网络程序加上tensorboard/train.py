import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
regulariztion_rate=0.0001
moving_average_decay=0.99
learning_rate_base=0.8
learning_rate_decay=0.99
Batch_size=100
training_steps=1000

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")
    regularizer=tf.contrib.layers.l2_regularizer(regulariztion_rate)
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)


    with tf.name_scope("moving_average"):
        variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        variable_averages_op=variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_function"):
        cross_entry=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y_,1),logits=y)
        cross_entry_mean=tf.reduce_mean(cross_entry)      #cross_entry每一行是每个样本的损失值
        loss=cross_entry_mean+tf.add_n(tf.get_collection("losses"))

    with tf.name_scope("train_op"):
        #学习率衰减
        learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/Batch_size,
                                                 learning_rate_decay,staircase=False)
        #global_step=global_step不能少，保证其不断变化，然后
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        with tf.control_dependencies([train_step,variable_averages_op]):
            train_op=tf.no_op(name="train")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('log/train', sess.graph)
        for i in range(training_steps):
            xs,ys=mnist.train.next_batch(Batch_size)

            #每100步轮记录一次运行状态
            if i %100==0:
                #配置运行时需要记录的信息
                run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #运行时记录运行信息的proto
                run_metadata=tf.RunMetadata()

                #将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时的每一个节点的时间和空间开销等信息
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                                               options=run_options,run_metadata=run_metadata)
                #达到一定训练次数后，记一次meta做一下标记,这样可以每迭代100次记录一个当前计算图的状态
                train_writer.add_run_metadata(run_metadata,"step%03d"%i)   #保证长度至少为3,用0补齐
                print("经过%s轮后，模型总损失值为%s"%(step,loss_value))
            else:
                    #因为在GradientDescentOptimizer中设定参数global_step=global_step，所以每运行一次train_step
                    #变量global_step就会自增一次

                    #正常运行，不记录运行信息
                 _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

        train_writer.close()
    writer=tf.summary.FileWriter("log",tf.get_default_graph())    #这里仅仅能看到整个计算图的结构而已，不能看到运行时的相关信息
    writer.close()                                                #计算图的运行信息在add_run_metadata中，被保存在"log/train"文件夹下



def main(argv=None):   #这里必须加argv=None，或者直接写个短下划线_
    mnist=input_data.read_data_sets("../MNIST_data",one_hot=True)
    # print(mnist.train.labels[3])
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




#生成的tensorboard图中，"Variable"节点表示的是global_step，它和"train_op"节点有一条双向的边表明其依赖关系
#因为两者是通过tf.control_dependencies联系在一起的