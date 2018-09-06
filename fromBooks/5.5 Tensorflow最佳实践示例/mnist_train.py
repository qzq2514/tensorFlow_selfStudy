import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import mnist_inference
import os

Batch_size=100
learning_rate_base=0.8
learning_rate_decay=0.99
regulariztion_rate=0.0001
training_steps=10000
moving_average_decay=0.99

model_save_path="Saver"
model_name="bestSample"



def train(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.Input_node],name="x-input")
    y_=tf.placeholder(tf.float32,[None,mnist_inference.Output_node],name="y-input")

    regularizer=tf.contrib.layers.l2_regularizer(regulariztion_rate)

    #前向传播得到最终预测值[m,Output_node]
    y=mnist_inference.inference(x,regularizer)


    #开始计算损失值
    global_step=tf.Variable(0,trainable=False)    #保存当前迭代的次数

    variable_average=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_average_op=variable_average.apply(tf.trainable_variables())    #将所有可训练的变量进行滑动平均(包括权重和偏置)

    #添加softmax
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)

    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))
    #学习率衰减
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/Batch_size,
                                             learning_rate_decay)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_average_op]):
        train_op=tf.no_op(name="train")

    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(training_steps):
            xs,ys=mnist.train.next_batch(Batch_size)
            #因为在GradientDescentOptimizer中设定参数global_step=global_step，所以每运行一次train_step
            #变量global_step就会自增一次
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if i %1000==0:
                print("经过%s轮后，模型总损失值为%s"%(step,loss_value))


                #这里加入了global_step参数，使得每次保存的文件都能加上轮数作为后缀
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)


def main(argv=None):
    mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
    # print(mnist.train.labels[3])
    train(mnist)

if __name__ == '__main__':
    tf.app.run()


