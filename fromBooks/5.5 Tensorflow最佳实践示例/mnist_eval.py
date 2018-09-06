import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

#每十秒加载一次最新的模型，并在测试集上测试最新的模型准确率
eval_interval_secs=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_inference.Input_node],name="x-input")
        y_=tf.placeholder(tf.float32,[None,mnist_inference.Output_node],name="y-output")

        value_feed={x:mnist.validation.images,y_:mnist.validation.labels}

        #测试时不用关心正则化的损失
        y=mnist_inference.inference(x,None)

        correct_pred=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        variable_average=tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        variable_restore=variable_average.variables_to_restore()


        saver=tf.train.Saver(variable_restore)   #variable_restore是一个字典集合，键是"变量name"/ExponentialMovingAverage形式，值是同name的变量

        while True:
            with tf.Session() as sess:
                #get_checkpoint_state根据该目录下的checkpoint文件自动找到最新模型的文件名
                fi=tf.train.get_checkpoint_state(mnist_train.model_save_path)

                if fi and fi.model_checkpoint_path:
                    #checkpoint文件第一行就是:model_checkpoint_path: "bestSample-5001"
                    #print("Path:",fi.model_checkpoint_path)       #输出：Saver/bestSample-5001

                    #在调用inference函数后，就已经生成了每层对应的权重，且是和保存的模型中是同名的
                    #之后再调用restore就会把保存的模型中的变量值赋值给同名的变量(所以这里不能再修改变量的name)
                    saver.restore(sess,fi.model_checkpoint_path)

                    global_step=fi.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    #模型的全部参数被加载赋值好后,就能得到模型在验证集的准去旅
                    accuracy_score=sess.run(accuracy,feed_dict=value_feed)
                    print("经过%s轮后，模型在验证集上的精确度为%s" % (global_step, accuracy_score))

                else:
                    print("未找到checkpoint文件")
                    return
                time.sleep(eval_interval_secs)


def main(argv=None):
    mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
    # print(mnist.train.labels[3])
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()




