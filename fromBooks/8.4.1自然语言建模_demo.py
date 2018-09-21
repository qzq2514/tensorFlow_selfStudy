import numpy as np
import tensorflow as tf
#默认没有以下的模块，会出问题，自己到github上下载指定的reader.py文件和__init__.py文件，然后放在tensorflow库目录下
#详见：https://www.jianshu.com/p/65490010a485
from tensorflow.models.rnn.ptb import reader
Data_Path="ptbData/simple-examples/data"

train_data,valid_data,test_data,_=reader.ptb_raw_data(Data_Path)

# print(len(train_data))
# print(train_data[:20])  #训练集是一个长929589的句子，每个ID代表一个单词，其中ID为2代表结束符号

#这里将原始数据得到4个batch,每个batch有5个序列元素
batch=reader.ptb_producer(train_data,4,5)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #因为ptb_producer中有tf.train.range_input_producer函数，必须先启动线程才能正确运行,
    #而tf.train.range_input_producer(NUM_EXPOCHES, num_epochs=1, shuffle=False)会
    #产生一个队列，队列包含0到NUM_EXPOCHES-1的元素，num_epochs表示迭代轮数
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        x,y=sess.run(batch)
        #ptb_producer返回的是数据和对应的标签，因为自然语言模型预测下一个值，所以标签即为该单词的后面一个单词
        print("X:",x)
        print("y:",y)

    # 关闭多线程
    coord.request_stop()
    coord.join(threads)